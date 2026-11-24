# Schemas and compatibility

Every machine-readable AHL document carries a schema id. v0.1 freezes these ids; a change in
meaning requires a **new** id, never a silent edit.

| Schema id | Where | Persisted? | Status |
|---|---|---|---|
| `ahl-runplan-v0` | `RunPlan`; embedded as `plan` in `run.json`; printed by `ahl run --dry-run --json` | Yes, inside `run.json` | **Transitional** (`v0`): the shape may still grow. |
| `ahl-run-v1` | `run.json` | Yes | **Canonical run record.** |
| `ahl-metadata-v1` | `metadata.json` next to `run.json` | Yes | Derived convenience copy; `run.json` is the source of truth. |
| `ahl-evaluation-v1` | `evaluation.json` | Yes | **Canonical evaluation record.** |
| `ahl-metrics-v1` | trace/patch-derived metrics (`ahl results`, `ahl compare`, experiment summaries) | **No**: computed on demand from `trace.jsonl` and `model.patch` | Stable semantics. |
| `ahl-experiment-v1` | the experiment YAML you write | Yes (byte copy + normalized `experiment.json`) | **Public input format.** |

Experiment-directory documents have their own ids: `ahl-experiment-record-v1`
(`experiment.json`), `ahl-matrix-v1` (`matrix.json`) and `ahl-experiment-summary-v1`
(`summary.json`).

## `ahl-runplan-v0`

The fully resolved, secret-free description of one run: benchmark, task (with task-repository
commit and dirty flag), prompt (sha256 of the exact text; the text itself is `prompt.txt`),
harness (path, version, expected pin), invocation (argv, non-secret env, secret env **names**,
generated config files), provider, model and reasoning, timeout, network plan, AHL provenance
(commit, dirty, version) and, for experiment runs, `experiment {experiment_id,
definition_sha256, matrix_cell_id, repetition, attempt}`. `purpose` is `canary` or `experiment`.
New optional fields are added without a version bump while the `v0` suffix stands; readers must
ignore nothing silently (the models are strict), so treat `v0` as "may still change".

## `ahl-run-v1`

State (`prepared | running | completed | harness_failed | timed_out | failed`), `failed_stage`,
timings, exit code, the embedded plan, the image actually used (reference, **image id**, repo
digests), container identifiers, provisioning, network (including the Squid policy hash),
SHA-256 hashes of the prompt, patch, trace and generated config, the cleanup result and
warnings. Written atomically at the start, when the harness starts, and at the end.

## `ahl-evaluation-v1`

`state`, `resolved` (`true`/`false`/`null`), `classification`, evaluator version, the patch hash,
task-repository commit, the three image identities (agent image, evaluator-built image, restored
tag), the evaluator-image baseline verification, build-input hashes, the official result for the
instance, and any AHL-decided `reason`. Only `resolved` true/false is a benchmark verdict;
`null` means no verdict and is excluded from denominators.

## `ahl-metrics-v1`

Per run: model turns; tool calls and tool errors; read/bash/edit/write/other calls;
verification calls; input, output, reasoning, cache-read and cache-write tokens; cost; and patch
statistics (files, test/non-test files, insertions, deletions). Semantics are those of the frozen
research extractor and reproduce its published CSV exactly. A metric that a trace cannot supply
is `null`/`N/A`, never zero. The verification-command heuristic is a text match and undercounts
some runners (see `limitations.md`).

## `ahl-experiment-v1`

See [experiments.md](experiments.md). Unknown fields are rejected.

## Compatibility expectations (v0.1)

* **Read compatibility:** `ahl` reads `ahl-run-v1` and `ahl-evaluation-v1` documents written by
  any 0.1.x version. Historical research runs (several `metadata.json` shapes and a flag-style
  `evaluation.json`) are readable through a **read-only** legacy loader; AHL never writes them.
* **Within 0.1.x** documents may gain optional fields; existing fields keep their meaning.
* **Breaking changes** get a new schema id and keep the old loader.
* `ahl-runplan-v0` and the internal `ahl-experiment-record-v1` / `ahl-matrix-v1` /
  `ahl-experiment-summary-v1` are the most likely to evolve.
