# Experiments and benchmark matrices

`ahl benchmark EXPERIMENT.yaml` turns a versioned experiment definition into a matrix of runs,
pre-flights **all** of it, then runs and evaluates the cells one after another.

The experiment file says **what** to benchmark. How one run (`docs/execution.md`) or one
evaluation (`docs/evaluation.md`) works is unchanged: the orchestrator builds ordinary `RunPlan`s,
calls the generic executor and the standalone evaluation service, and never contains
harness-specific or evaluator-specific logic.

```
experiment.yaml -> validate -> normalize (+ hash) -> matrix -> RunPlans
                -> experiment-wide pre-flight -> executor -> evaluator -> summary
```

## `ahl-experiment-v1`

```yaml
schema_version: ahl-experiment-v1
experiment:
  name: astra-comparison        # sanitized into the experiment id
  description: ...
  purpose: comparison
benchmark:
  name: swebench-multilingual
  task_repository_commit: 6e08cbcd763c869abd7aae4bcb9e1b3991180777   # strongly recommended
tasks:                           # an explicit list ...
  - babel__babel-15445
  # ... or a task-set file (relative to this file):  tasks: {file: ../configs/challenge-set-v1.json}
harnesses: [pi, opencode]
models:
  - {provider: openai, model: gpt-6-astra, reasoning: medium}
repetitions: 1
execution: {timeout_seconds: 3600, network: restricted-provider}
evaluation: {enabled: true}
```

* Unknown fields are rejected (so `api_key:` can never be silently accepted). Credentials stay in
  the environment variables named by the provider (`docs/run-plan.md`).
* Task-set files accept `{"tasks": [...]}` or the `ecosystems` layout of
  `configs/challenge-set-v1.json`. The file's SHA-256 is part of the experiment hash.
* No automatic task selection by difficulty exists.

### Normalization, hash and identity

The definition describes a **set**: tasks, harnesses and models are sorted, defaults filled in,
task-set files resolved. The SHA-256 of that canonical JSON is the *definition hash*; comments,
whitespace and list order do not change it. The experiment id is
`<sanitized-name>-<UTC start>-<first 8 hex of the hash>`, e.g.
`astra-comparison-20261124T180000Z-a31c82f4`.

## Matrix

`tasks x harnesses x models x repetitions`, expanded in the sorted order task, harness, model,
repetition. A cell id is a pure function of its coordinates:
`task=<id>;harness=<name>;model=<provider/model>;reasoning=<level|none>;repetition=<n>`.

## Pre-flight (all cells, before the first run)

Every cell gets a RunPlan and the same static checks as `ahl run --preflight` (secrets present,
task image and proxy image local, provisioning sources exist, ...), plus:

* the task repository equals `task_repository_commit` (exactly) and is **clean**; it is never
  checked out or modified. An unpinned experiment is allowed but flagged;
* every task resolves; every harness exists; harness versions equal the repository pins;
* when evaluation is enabled: the evaluator environment imports and every task has its evaluator
  files (existence only).

Failures from all cells are reported together; **any** failure means zero runs, and no
experiment directory is created. Development-only overrides exist as CLI flags (never in the
file): `--allow-dirty-task-repo`, `--allow-harness-version-mismatch`. They are printed
prominently and recorded in `experiment.json` and `summary.json`.

## Dry run

`ahl benchmark EXPERIMENT.yaml --dry-run [--json]` validates, verifies the pin, expands the
matrix, resolves every RunPlan, runs the static pre-flight (read-only `docker image inspect`
only) and prints the plan: counts, a worst-case duration estimate, every cell and every
failure. It creates nothing, executes nothing, evaluates nothing, and needs neither the gate nor
secrets (missing secrets are *reported*).

## Real execution gate

Unchanged: `AHL_ENABLE_REAL_EXECUTION=1` (exactly `1`) is required. Without it
`ahl benchmark` refuses before touching Docker or creating any file. There is no bypass flag, and
the executor re-checks the gate for every run.

## Artifacts

```
results/experiments/<experiment-id>/
  experiment.yaml   byte-for-byte copy of the file you launched
  experiment.json   id, hash, start/finish, overrides, AHL + task-repo provenance, normalized definition
  matrix.json       every cell, its state and its attempts (rewritten atomically on every transition)
  summary.json      recomputable summary (rewritten atomically after every cell)
```

Raw runs stay where they always were (`results/raw/swebench/<task>/<run-id>/`). Each run's
`run.json` records `purpose: experiment` and `experiment: {experiment_id, definition_sha256,
matrix_cell_id, repetition, attempt}`. All writes are atomic.

## Cell state machine

```
planned -> running -> completed          (agent OK, evaluation disabled)
                   -> agent_completed -> passed | failed | no_verdict
                   -> harness_failed    (harness exit != 0; not evaluated)
                   -> timed_out         (not evaluated)
                   -> infra_failure     (execution infrastructure failed, or interrupted)
```

**Evaluation rule.** Only a cleanly completed agent run (exit 0, patch captured) is evaluated
automatically. After a harness failure or timeout the patch is kept as evidence, but the cell is
an agent outcome, not a benchmark verdict, and is excluded from the correctness denominator. A
person can still score the preserved patch with `ahl evaluate <run-id>`.
`no_verdict` (evaluator infrastructure/ambiguous result) is never reported as FAIL.

## Resume

`ahl benchmark EXPERIMENT.yaml --resume <experiment-id>` (the same file; a changed definition is
refused). State is **derived from canonical records**, not from `matrix.json` or file names: a
cell is complete only if its `run.json` validates, names this experiment/cell/attempt, and (for
evaluated cells) a valid `evaluation.json` exists.

* completed and evaluated cells never rerun;
* a completed run without a valid evaluation resumes at evaluation (the agent is not rerun; a
  broken/partial evaluation is archived as `evaluation.superseded-*`, never deleted);
* a run interrupted midway is an `infra_failure`; failed/infrastructure/timeout cells are **not**
  rerun unless `--retry-failed`, which creates attempt N+1 (a new run directory) and keeps every
  earlier attempt untouched. `no_verdict` cells are re-evaluated instead of rerun, so a flaky
  evaluator never costs another agent run.

## Sequential execution

Cells run one at a time, in matrix order. The official evaluator rebuilds and re-tags the shared
task image (`docs/evaluation.md`), so concurrent evaluations would race on Docker tags; model
rate limits and shared task images add further hazards. Concurrency can be added later without
changing the artifact format.

## Summary

`summary.json` (and `ahl experiments show <id>`, which works offline without Docker, network or
API keys) reports planned/executed/evaluated cells, PASS, FAIL, no verdict, infrastructure
failures, harness failures, timeouts, agent and evaluation durations, and cost, turns, tool calls
and token metrics, overall and grouped by harness and by model. Correctness is shown as
`resolved / evaluated` with no-verdict cells excluded. Aggregation is the same code as
`ahl compare`; unavailable metrics stay `null` (never zero). Only the latest attempt of a cell
counts. No statistical significance is claimed.

## Examples

* `examples/minimal.yaml`: one task, one harness, one model.
* `examples/astra-comparison.yaml`: the 11-task Pi vs OpenCode design on GPT-6 Astra. It is
  **not** the original study and imports none of its results; running it would create new runs.
