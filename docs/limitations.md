# Known limitations (v0.1)

* **Harnesses:** only Pi and OpenCode.
* **Benchmark:** only SWE-bench Multilingual (task discovery and the evaluation backend).
* **Sequential experiments:** cells run one at a time (no concurrency; the evaluator re-tags a
  shared Docker image, and provider rate limits are not modelled).
* **No automatic Docker image acquisition:** task and proxy images must already exist locally;
  a missing image is a pre-flight failure.
* **External setup required:** the task repository and a SWE-bench evaluator environment
  (`swebench` 5.0.2 tested) are obtained and configured by you.
* **Evaluator image handling:** the official evaluator force-rebuilds the task image (several GB,
  minutes per task). AHL snapshots and restores your image tag and removes the evaluator-built
  image unless `--keep-evaluator-image` is given. The evaluator image differs from the agent
  image; its baseline commit is verified, not its whole content.
* **Nondeterminism:** model outputs vary between runs (see `reproducibility.md`).
* **Cost metrics:** provider-reported cost can be meaningless for self-hosted or custom
  OpenAI-compatible endpoints (often zero or an artefact of the harness's own price table).
* **`ahl-metrics-v1` verification heuristic:** a text match for test runners; it undercounts
  runners it does not list (for example `mvn test`, `./gradlew test`, `make test`), so Java tasks
  are likely under-counted. The semantics are frozen for comparability with the original study.
* **Legacy support is read-only:** historical research metadata and evaluation files can be read,
  never written.
* **Execution gate:** `AHL_ENABLE_REAL_EXECUTION=1` is a temporary pre-release safety gate and
  may change.
* **Platforms:** Linux and WSL2 only.
* **Single-pass runs:** the research "verification" two-pass protocol is not representable as a
  RunPlan.
* **Statistics:** AHL reports counts and rates and claims no statistical significance.
