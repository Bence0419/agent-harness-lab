# Evaluation

Evaluation is a separate, post-run step: `ahl evaluate RUN_ID` scores a run's persisted
`model.patch` with the benchmark's official evaluator. It calls no model, starts no agent and
needs neither `OPENAI_API_KEY` nor `AHL_ENABLE_REAL_EXECUTION`. It is deliberately not part of
`ahl run`.

## Interface

`EvaluationAdapter` (name, benchmark, `preflight(ctx)`, `evaluate(ctx, preflight)`) is looked up
by benchmark name. The input (`RunSubject`) contains only benchmark, instance, image, task-repo
commit and patch path, so Pi and OpenCode runs take the identical path and no evaluation code
imports or knows a harness.

## Configuration

* task repository: `--tasks-repo` / `AHL_SWEBENCH_TASKS_REPO`
* evaluator environment: `--swebench-python` / `AHL_SWEBENCH_PYTHON` (the interpreter of an
  environment that has `swebench` installed). The evaluator's console script next to it is run
  under that interpreter. The detected Python and `swebench` versions are recorded.
* `--preflight` checks everything without evaluating; `--output-dir` writes elsewhere (used to
  re-check historical runs without touching them); `--force` archives and re-evaluates.

## Evaluator-side data boundary

`benchmarks/evaluator_data.py` is the only AHL code that touches `eval.sh`, `test.patch`,
`tests.json` (hashed for provenance), `Dockerfile` and `gold.patch` (existence only). The ordinary
`Task` and its adapter never import it. `hints.md` is never read.

## `evaluation.json` (`ahl-evaluation-v1`)

`run_id, benchmark, evaluator, evaluator_version, task, started_at, finished_at,
duration_seconds, state, resolved, classification, patch_sha256, task_repository_commit,
run_task_repository_commit, image {reference, image_id_before, image_id_after}, environment,
command {argv, cwd, eval_run_id, exit_code, timed_out}, evaluator_inputs_sha256, official_result,
artifacts {path: sha256}, error, warnings, replaces`. Evaluation time is never merged into the
agent's duration. Written atomically; an existing file is refused unless `--force`, which archives
the old file (`evaluation.superseded-<ts>.json`) and its artifacts.

## Verdicts

| official report | state | `resolved` | classification |
|---|---|---|---|
| resolved (completed, exit 0) | `resolved` | true | benchmark_pass |
| unresolved | `unresolved` | false | benchmark_fail |
| empty patch | `empty_patch` | false | benchmark_fail |
| infra failure / no report / timeout | `infra_failure` | **null** | evaluation_failure |
| evaluator `error` / `incomplete` | `error` / `incomplete` | **null** | evaluation_failure |
| several lists / none / "resolved" with failing exit | `ambiguous_failure` / `unknown` | **null** | ambiguous |

`null` means "no verdict": excluded from correctness denominators, never counted as a FAIL.
The same mapping is applied to historical evaluation files (all 49 are resolved/unresolved).

## The evaluator rebuilds task images (swebench 5.0.2) and how AHL contains it

Verified in the evaluator's source: with `--task-repo` it loads instances and tests from the
local task repo and calls `_build_before_eval` -> `build_instance_images(force_rebuild=True)`,
which removes the image tag named in `task.yaml` and builds a NEW image under it. `swebench eval`
has no existing-image mode and no namespace/tag option, and omitting `--task-repo` would switch
the test source to the published HuggingFace dataset (different semantics, needs network). Each
rebuild yields a different image id. So the evaluator is run unchanged and AHL guards the tag:

1. pin the current image id under `ahl-eval-snapshot/<name>:<run>-<ts>` (the evaluator removes by
   tag name, so the pin keeps the original image alive); if pinning fails, nothing is evaluated;
2. run the evaluator; 3. re-tag the original id onto the task tag and VERIFY it; only then drop the pin;
4. verify the evaluator image's baseline (below), then release the evaluator image. If restoration
   cannot be verified the pin is kept, `restoration: failed` is recorded, a `TASK IMAGE TAG NOT
   RESTORED` warning with the exact `docker tag` recovery command is shown and the CLI exits 1.
This runs in `finally`, so it also holds after evaluator failure and Ctrl-C.

`evaluation.json` keeps four identities apart: `agent_image_id` (run.json), `image_id_before`
(tag at start), `evaluator_image_id` (what tests ran in), `final_image_id` (tag afterwards).
Because every official evaluation rebuilds, two evaluations (e.g. a Pi and an OpenCode patch)
do NOT share an evaluator image id; both are recorded so this is visible.

### Evaluator image baseline (gates every model-attributable verdict)

A patch-apply failure (or any verdict) can only be charged to the model if the rebuilt image
holds the task's baseline. After the official run, AHL inspects that exact image BY ID (never the
restored shared tag) in a throwaway, network-less container and checks: `/testbed` exists, is a
git repository, `HEAD == Task.base_commit`, and `git status --porcelain` is clean (with
`core.fileMode=false`, as in the frozen workspace preparation). It reads environment provenance
only. The structured result is `evaluator_image_baseline` {verification_status
(`verified|mismatch|dirty|unverified`), image_id, expected_base_commit, actual_base_commit,
testbed_exists, is_git_repository, worktree_clean, reason, error}.

If it is not `verified`, a `resolved`, `unresolved` or `patch_apply_failed` result becomes
`infra_failure` (`resolved: null`, excluded from correctness denominators) with `reason`
`evaluator_image_baseline_mismatch|_dirty|_unverified`; the original official state is kept in
`official_result.state_before_baseline_check`. `empty_patch` is unaffected (nothing to apply).
`evaluator_build_inputs_sha256` records the SHA-256 of the task's `Dockerfile` and `task.yaml`
(supplementing, not replacing, the task repository commit).

### Evaluator image retention

By default the image the evaluator built is REMOVED after the baseline check, with a plain
`docker rmi <id>` (Docker refuses it while any tag references it) and only if its id did not
exist before the evaluation; there is no pruning and the restored original is never touched.
`--keep-evaluator-image` keeps it under `ahl-eval-image/<name>:<id12>` for post-mortems (and
records the request). If tag restoration failed, nothing is cleaned up (`retained-for-recovery`).
