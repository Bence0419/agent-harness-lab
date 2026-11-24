# Research State v1 (frozen)

## Purpose

This document describes the research state of the repository before it was
turned into the `ahl` CLI toolkit. The Bash runners, configs, experiment
manifests and result summaries listed here produced the completed studies and
remain the source of truth for them. They are not to be altered in behavior;
the new Python package is built beside them.

## Pinned components

| Component | Version |
|---|---|
| Pi coding agent (`@earendil-works/pi-coding-agent`) | 1.0.2 |
| OpenCode (`@opencode/cli`) | 2.2.0 |
| SWE-bench (evaluator) | 5.0.2 |
| SWE-bench Multilingual task repository | `SWE-bench/swe-bench-multilingual-tasks` @ `6e08cbcd763c869abd7aae4bcb9e1b3991180777` |
| Squid proxy image | `ubuntu/squid@sha256:6a097f68bae708cedbabd6188d68c7e2e7a38cedd05a176e1cc0ba29e3bbe029` |

The task-repository commit is enforced by every SWE-bench runner, which aborts
on mismatch.

## GPT-6 Astra comparison

- Model: `gpt-6-astra` (OpenAI, reasoning effort `medium`; a model alias, not a
  dated snapshot).
- Tasks: 11 SWE-bench Multilingual instances: the 5 pilot tasks plus the 6 of
  `configs/challenge-set-v1.json`.
- Harnesses: Pi and OpenCode, each with its default behavior (see
  `configs/pi-frontier-astra-v1.json`, `configs/opencode-frontier-astra-v1.json`).
- Runs: 11 tasks x 2 harnesses = **22 runs**, one attempt each.
- Results (official SWE-bench evaluation):
  - Pi: **11/11** resolved
  - OpenCode: **10/11** resolved (unresolved: `babel__babel-15445`)
- Derived metrics: `results/analysis/astra-trajectory-metrics.csv`, produced by
  `scripts/extract-trajectory-metrics.py`.

## Runners that produced the frozen results

| Runner | Used for |
|---|---|
| `scripts/run-swebench-task.sh` | Pi baseline, gpt-5.4-mini (no network policy) |
| `scripts/run-swebench-task-verification.sh` | Pi + verification pass (verification-v1) |
| `scripts/run-swebench-task-opencode.sh` | OpenCode reference, gpt-5.4-mini |
| `scripts/run-swebench-task-astra.sh` | Pi, GPT-6 Astra |
| `scripts/run-swebench-task-opencode-astra.sh` | OpenCode, GPT-6 Astra |
| `scripts/run-swebench-task-bme-qwen.sh` | Pi, OpenAI-compatible Qwen endpoint |
| `scripts/run-swebench-task-opencode-bme-qwen.sh` | OpenCode, OpenAI-compatible Qwen endpoint |
| `scripts/evaluate-swebench-run.sh` | Official evaluation of a run directory |
| `scripts/run-baseline.sh`, `summarize-*.js` | Earlier JS bug-fix baseline (legacy) |

## Raw artifacts are not in Git

`results/raw/` is gitignored on purpose (traces, patches, logs, evaluator
output; about 180 MB). Integrity can be recorded with
`scripts/create-research-archive-manifest.py`, which writes a SHA-256 manifest
without copying or modifying anything. Publishing the raw data, if desired, is
a separate decision.

## Reproducibility meaning

Runs are non-deterministic: LLM output varies between runs and model aliases
can change server-side. Reproducibility here means **reproducing the
experimental configuration** (harness version, model, task set, task-repo
commit, container image, network policy, prompts), not obtaining byte-identical
patches or traces.

## Known limitations

- Some older experiments predate the unified network policy. Pi runs with
  gpt-5.4-mini ran without the egress allowlist, and the Pi runners do not
  record `networkPolicy` in `metadata.json`. These runs are not directly
  comparable to the Astra and OpenCode runs under identical network
  conditions.
- `metadata.json` has several shapes (Pi, OpenCode, verification).
- Some config files originally referenced an institutional OpenAI-compatible endpoint. The
  public repository contains a hostname-redacted representation of them (the host is
  replaced by `llm.example.org`); see `docs/public-release.md`. The research
  methodology and results are unchanged.
