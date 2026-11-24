# Agent Harness Lab (AHL)

AHL is a command-line toolkit for benchmarking AI **coding-agent harnesses** (the program that
drives a model through a coding task) reproducibly. It plans a run, executes the harness in an
isolated Docker container, captures the resulting patch and trace, scores the patch with the
benchmark's official evaluator, and keeps detailed provenance so the experiment can be
described, repeated and audited.

## The problem it solves

Comparing coding agents is easy to get subtly wrong: different harness versions, models,
prompts, network access, container images or task checkouts quietly change the result, and the
differences are rarely written down. AHL makes each of those an explicit, recorded input,
refuses to start an experiment whose configuration does not validate, and separates *the model
failed the task* from *the infrastructure failed*.

## What is supported (v0.1)

| | Supported |
|---|---|
| Benchmark | SWE-bench Multilingual (task repository + official evaluator, both obtained separately) |
| Harnesses | [Pi](https://www.npmjs.com/package/@earendil-works/pi-coding-agent) and OpenCode |
| Providers | OpenAI, and generic OpenAI-compatible endpoints |
| Platform | Linux and WSL2 |

See [Limitations](#limitations) for what is deliberately not there yet.

## Reproducibility model

AHL reproduces the **experimental configuration** and preserves **detailed provenance**: harness
and version, model and reasoning level, task and task-repository commit, container image id,
network policy, prompt hash, generated configuration hashes, AHL version/commit and evaluator
version. **LLM outputs themselves remain nondeterministic**: the same configuration can produce
different patches, and a provider can change a model alias server-side. AHL does not claim
reproducible model output. Details: [docs/reproducibility.md](docs/reproducibility.md).

## Architecture in one picture

```
task repository ──> Task ──> RunPlan (ahl-runplan-v0) ──> executor ──> run.json (ahl-run-v1)
providers + harness adapters ─┘          (Docker, no harness logic)     model.patch, trace.jsonl
                                                                              │
experiment.yaml ──> matrix ──> RunPlans ──> executor ──> evaluator ──> evaluation.json
 (ahl-experiment-v1)                        (sequential)   (official)   (ahl-evaluation-v1)
                                                                              │
                                              results / compare / experiments show (read-only)
```

Benchmarks, harnesses and evaluators sit behind small adapter interfaces; the executor knows
nothing about Pi or OpenCode. See [docs/architecture.md](docs/architecture.md).

## Requirements

AHL itself is a small Python package. Running real experiments needs several **external**
dependencies that AHL does **not** bundle:

| Needed for | Requirement |
|---|---|
| AHL itself | Python 3.12+, [uv](https://docs.astral.sh/uv/) (recommended), Git |
| Running agents | Docker (daemon running), Node.js 22.19+ on the host (`package.json` engines; 24.x is what was tested), Pi and/or OpenCode installed with npm |
| Tasks | a local checkout of the SWE-bench Multilingual task repository |
| Scoring | a separate Python environment with `swebench` installed (the tested version is 5.0.2) |
| Real runs | an API key for the model provider, in an environment variable |

Task Docker images must already exist locally; AHL never pulls or builds them (yet).

## Installation

### 1. AHL

From a clone (the supported development workflow):

```bash
git clone <repository-url> agent-harness-lab && cd agent-harness-lab
uv sync
uv run ahl version      # Agent Harness Lab 0.1.0
uv run ahl doctor       # checks the environment; warnings are normal on a fresh machine
```

As a package: `uv build` creates a wheel and sdist in `dist/`; `pip install dist/*.whl` installs
the `ahl` command. Nothing is published to PyPI.

### 2. Harnesses (optional, for real runs)

Install Pi and OpenCode with npm, either project-local or globally. A source checkout pins the
tested versions in `package.json` (Pi 1.0.2, OpenCode 2.2.0): `npm ci` installs them into
`node_modules/`, which `ahl` finds automatically. An installed (wheel) AHL finds them on `PATH`.
For a reproducible experiment the installed versions must equal the pins in a source checkout;
otherwise pre-flight fails.

### 3. Task repository

```bash
git clone <swe-bench-multilingual-tasks-url> /path/to/swe-bench-multilingual-tasks
git -C /path/to/swe-bench-multilingual-tasks checkout <pinned-commit>
export AHL_SWEBENCH_TASKS_REPO=/path/to/swe-bench-multilingual-tasks
```

### 4. Evaluator environment

```bash
uv venv /path/to/swebench-venv && uv pip install --python /path/to/swebench-venv/bin/python swebench==5.0.2
export AHL_SWEBENCH_PYTHON=/path/to/swebench-venv/bin/python
```

## Quick start

All of these are read-only until the "real run" step.

```bash
export AHL_SWEBENCH_TASKS_REPO=/path/to/swe-bench-multilingual-tasks
export OPENAI_API_KEY=...                         # only needed for a real run

uv run ahl doctor                                 # environment diagnostics
uv run ahl tasks list --repo babel/babel          # discover tasks
uv run ahl tasks show babel__babel-15445
uv run ahl run --task babel__babel-15445 --harness pi \
    --model openai/gpt-6-astra --reasoning medium --dry-run
```

### Dry-run and pre-flight

* `--dry-run` resolves and prints the complete plan (task, harness and version, model, prompt
  hash, network policy, required secret *names*). It touches no Docker and writes nothing.
* `--preflight` additionally checks that the files to mount exist, secrets are *present*, and the
  task and proxy images exist locally, then prints `READY` or `NOT READY`. It creates nothing.

### A single run

Real execution calls the model API and **costs money**. It is disabled unless you opt in:

```bash
AHL_ENABLE_REAL_EXECUTION=1 uv run ahl run --task babel__babel-15445 --harness pi \
    --model openai/gpt-6-astra --reasoning medium
```

`AHL_ENABLE_REAL_EXECUTION=1` is a temporary pre-release safety gate (exactly `1`; there is no
bypass flag). The run directory is created under `results/raw/swebench/<task>/<run-id>/`.
Single runs are recorded with purpose `canary`; use experiments for comparisons.

OpenAI-compatible endpoints are declared in a providers file (credentials stay in environment
variables, named by `api_key_env`):

```yaml
# providers.yaml  (select with --providers-file or AHL_PROVIDERS_FILE)
providers:
  my-lab:
    type: openai-compatible
    base_url: https://llm.example.org/v1      # or base_url_env: MY_LAB_BASE_URL
    api_key_env: MY_LAB_API_KEY
```

Then `--model my-lab/<model-id>`. See [docs/run-plan.md](docs/run-plan.md).

### Evaluation

```bash
uv run ahl evaluate <run-id> --preflight          # check, evaluate nothing
uv run ahl evaluate <run-id>                      # official SWE-bench scoring (no model call)
```

Evaluation needs Docker, `AHL_SWEBENCH_TASKS_REPO` and `AHL_SWEBENCH_PYTHON`, but no API key.
It writes `evaluation.json` next to the run. The official evaluator rebuilds the task image;
AHL restores your image tag afterwards and verifies the rebuilt image's baseline commit before
trusting a verdict. Only a verified evaluation counts as pass/fail. See
[docs/evaluation.md](docs/evaluation.md).

### Inspecting results

```bash
uv run ahl results                                # runs found under results/raw
uv run ahl results --harness Pi --json
uv run ahl compare --harness Pi --harness OpenCode
```

Success rates are `resolved / evaluated`; runs without a benchmark verdict are excluded from the
denominator, and unavailable metrics show `N/A` (never zero). Differences are descriptive; AHL
claims no statistical significance.

## Experiments

An experiment file declares **what** to benchmark; AHL expands it to a matrix of runs and
pre-flights **all** of it before starting any run.

```yaml
schema_version: ahl-experiment-v1
experiment: {name: my-comparison, purpose: comparison}
benchmark:
  name: swebench-multilingual
  task_repository_commit: <40-hex commit>      # the checkout must match exactly and be clean
tasks: [babel__babel-15445, axios__axios-5892]
harnesses: [pi, opencode]
models:
  - {provider: openai, model: gpt-6-astra, reasoning: medium}
repetitions: 1
execution: {timeout_seconds: 3600, network: restricted-provider}
evaluation: {enabled: true}
```

```bash
uv run ahl benchmark experiment.yaml --dry-run                    # validate + plan + pre-flight
AHL_ENABLE_REAL_EXECUTION=1 uv run ahl benchmark experiment.yaml  # run sequentially, evaluate
uv run ahl experiments show <experiment-id>                       # offline status and summary
```

Examples: [examples/minimal.yaml](examples/minimal.yaml) and
[examples/astra-comparison.yaml](examples/astra-comparison.yaml) (a *new* execution of the
design of the original study; running it creates new runs).

### Resuming

```bash
AHL_ENABLE_REAL_EXECUTION=1 uv run ahl benchmark experiment.yaml --resume <experiment-id>
```

State is rebuilt from the canonical `run.json` / `evaluation.json` files, so completed and
evaluated cells never rerun and a finished-but-unevaluated run resumes at evaluation. Failed
cells are not retried unless you add `--retry-failed`, which creates a new attempt and keeps the
old evidence. Details: [docs/experiments.md](docs/experiments.md).

## Output layout

```
results/raw/swebench/<task>/<run-id>/       one run (gitignored)
    run.json  metadata.json  prompt.txt  trace.jsonl  model.patch  git-status.txt
    <harness>.stderr.log  proxy.log  generated/  evaluation.json  evaluator/
results/experiments/<experiment-id>/        one experiment (gitignored)
    experiment.yaml  experiment.json  matrix.json  summary.json
```

`results/raw/` and `results/experiments/` are gitignored; the paths are relative to the working
directory (a source checkout uses its own `results/raw`). Archive a finished study with the
experiment definition, a SHA-256 manifest and the AHL git commit/tag:
see [docs/reproducibility.md](docs/reproducibility.md).

## Security model

Agent runs execute **untrusted, model-generated commands** inside a Docker task container.
Restricted-provider networking limits egress to the model provider's host, secrets are passed
by environment-variable name only (never stored in plans, records or logs), and the task
directory with evaluator material is never mounted. This reduces risk; it is **not** a perfect
sandbox. Docker and the host mounts remain the security boundary. Read [SECURITY.md](SECURITY.md)
before running AHL on a sensitive machine.

## Limitations

Only Pi and OpenCode, only SWE-bench Multilingual, sequential experiments, no automatic Docker
image acquisition, external task/evaluator setup, and nondeterministic model output. The full,
honest list is in [docs/limitations.md](docs/limitations.md).

## Research provenance

AHL grew out of a controlled study comparing Pi and OpenCode on 11 SWE-bench Multilingual tasks.
That study **predates the `ahl` CLI**. It was produced by the Bash runners in `scripts/` with the
configurations in `configs/`, and it stays frozen separately: [docs/research-v1.md](docs/research-v1.md).
Three things must not be confused:

1. **Frozen research results**: the original study (`results/analysis/`, the raw archive that is
   deliberately not in this repository).
2. **Product canaries**: later single runs made with `ahl` to validate the toolkit (for example
   on `axios__axios-4738`). They are *not* part of the original study.
3. **New experiments**: anything produced with `ahl benchmark`.

The scripts remain for provenance; they are not the documented workflow.

## Development and testing

```bash
uv sync
uv run ruff check .
uv run pytest                  # no Docker, no API key, no network, no LLM call
uv run python tools/check_public_release.py   # secret / internal-value scan
```

Opt-in suites (need Docker and the external task/evaluator setup): `uv run pytest -m docker`
and the real-evaluator regression, described in [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

If you use AHL in academic work, please cite it using [CITATION.cff](CITATION.cff) (a draft: it
contains only project information that is actually known and no DOI or publication), and state
the AHL version, the task-repository commit and the harness and evaluator versions.

## License

Agent Harness Lab is licensed under the [Apache License, Version 2.0](LICENSE).
