# Reproducibility guarantees

## What AHL guarantees

AHL makes the **configuration** of an experiment explicit, validated and recorded:

* the experiment definition is normalized and hashed; the original file is copied byte for byte;
* the task repository must equal the pinned commit and be clean (a pre-flight failure otherwise);
* harness versions must equal the repository pins (a pre-flight failure otherwise);
* every run records the task, base commit, container **image id**, harness path and version,
  model, reasoning level, timeout, network policy and Squid policy hash, prompt hash and generated
  configuration hashes, AHL version/commit/dirty state, and its place in the experiment
  (`experiment_id`, cell id, repetition, attempt);
* the whole matrix is pre-flighted before the first run, and an interrupted experiment can be
  resumed from canonical records;
* evaluation records the evaluator version, the evaluator image identity, the verification of its
  baseline commit and the hashes of the build-defining task files;
* raw evidence is never overwritten: new attempts and re-evaluations get new files, the old ones
  are archived.

## What AHL does not guarantee

* **LLM output is nondeterministic.** The same configuration can yield different patches and
  results. Model aliases (for example `gpt-6-astra`) can change server-side.
* The evaluator image is **rebuilt** by the official evaluator and differs from the image the
  agent ran in. AHL verifies that the rebuilt image's `/testbed` is at the task's base commit and
  clean, and treats anything else as an infrastructure failure, but it cannot prove the images
  are otherwise identical.
* Provider-reported costs may be meaningless for self-hosted or custom endpoints.
* Network access is limited to the provider host; the provider itself is outside AHL's control.
* Reproducing the environment of another machine needs the same images, harness versions and
  evaluator version; AHL records them but does not install them.

For the original study the same meaning applies: see `research-v1.md`.

## Archiving a finished study

Raw artifacts are gitignored. To keep a study citable:

1. Archive `results/raw/` (or just the relevant run directories) and the experiment directory
   `results/experiments/<experiment-id>/` with `tar` or `zip`.
2. Record a SHA-256 manifest: `find <dir> -type f -print0 | sort -z | xargs -0 sha256sum >
   manifest.sha256`. The source checkout also has `scripts/create-research-archive-manifest.py`.
3. Keep the experiment definition (`experiment.yaml`) and the AHL git commit or tag used.
4. Note the harness and evaluator versions and the task-repository commit (all in
   `experiment.json` and each `run.json`).
5. Do not commit raw results to the source repository unless a tiny fixture is intentional.
