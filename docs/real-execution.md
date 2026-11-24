# Real execution and single runs

## The temporary safety gate

Real Pi/OpenCode execution costs money and calls an external API, so `ahl run` (without
`--dry-run` / `--preflight`) refuses unless the environment contains exactly

    AHL_ENABLE_REAL_EXECUTION=1

Any other value (`0`, `true`, `yes`, `on`, empty, `" 1"`, `01`, ...) does NOT enable it. There is
no command-line flag that bypasses it. The generic executor re-checks the gate itself, so the
refusal also holds for programmatic use. `--dry-run` and `--preflight` never need the gate.
This is a temporary pre-release safety behavior, kept in v0.1 for both `ahl run` and
`ahl benchmark`; it may be replaced by an explicit approval workflow later.

## Run purpose

Every run carries a `purpose` (RunPlan and `run.json`). Runs created by `ahl run` are
`canary` runs: single runs that belong to no named experiment (use `ahl benchmark` for
experiments, whose runs have purpose `experiment`). `ahl results` shows them
(`purpose: canary`); `ahl compare` excludes them unless `--include-canaries` is given, so they
cannot silently mix into historical or future experiment comparisons. Canaries must use the
`restricted-provider` network; unrestricted networking is refused at pre-flight.

## Pre-flight

`ahl run --preflight ...` is stricter than `--dry-run` and still creates nothing: it checks the
gate, task and provenance, harness executable and version, rendered configuration, the exact host
paths that would be mounted, secret *presence*, the network plan and (read-only) Docker image
inspection. Real execution runs the same checks first and stops before creating any Docker
resource if one fails.

## Choosing a task for a first run

`ahl tasks history <id>` reports whether a task has ever been run (any run directory, complete
or not, or membership of a frozen research manifest). `ahl tasks candidates` applies a fixed
rule, independent of any benchmark outcome: the lexicographically first UNSEEN tasks whose
Docker image is already local; if fewer than requested exist it says so and lists the next unseen
tasks flagged as missing their image. Only task metadata is read; run outcomes and the
evaluator-side task files are never opened.
