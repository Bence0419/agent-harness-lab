# Providers, harness adapters and the RunPlan

`ahl run --dry-run` resolves user intent into a **RunPlan** (`ahl-runplan-v0`): a secret-free,
JSON-serializable description of one run. It is not `run.json`; it is what the executor carries out and the plan embedded in every
`run.json`. Planning executes nothing: no Docker, no agent, no LLM call, no files written.

## Model references and providers

* `--model PROVIDER/MODEL` splits on the **first** slash, so model ids may contain slashes:
  `example-lab/Org/Model-7B-FP8` is provider `example-lab`, model `Org/Model-7B-FP8`.
* `openai` is built in (`OPENAI_API_KEY`). OpenAI-compatible providers are defined in a providers
  file (`--providers-file` or `AHL_PROVIDERS_FILE`; YAML or JSON) with exactly one of `base_url`
  or `base_url_env`, plus `api_key_env`. Credentials are only ever environment-variable
  **names**; a file that contains a key is rejected without echoing it.
* `--reasoning` is one generic value. Pi receives `--thinking <level>` (validated against Pi's
  levels); OpenCode receives `provider/model#<level>`. No harness syntax lives in the model spec.

## Container contract inherited from the frozen runners

Invocation `argv[0]` is the path **inside the task container**, as in the frozen runners.
The executor provisions: Pi at `/opt/pi-node_modules/.bin/pi` (project `node_modules` mounted,
host Node at `/opt/node24`); OpenCode at `/usr/local/bin/opencode`. Generated config is written
to its `container_path` (Pi: `/root/.pi/agent/models.json`; OpenCode: `/etc/opencode/opencode.json`).
The agent prompt is the final argv element, `<<AHL_PROMPT>>`, substituted as one element.

## Timeout and reasoning capability

* `timeout_seconds` is a positive integer (execution time limit) or `null` (**no timeout**, as in
  the gpt-5.4-mini Pi, gpt-5.4-mini OpenCode and Qwen OpenCode frozen runners). New CLI runs
  default to 3600 s; there is deliberately no `--no-timeout` flag. A `null` timeout is warned about.
* Requested reasoning (`ModelSpec.reasoning`) and model capability (`model_capabilities.
  reasoning_supported`) are separate. Capability is tri-state: `true` / `false` / unknown
  (`null`), and is never guessed. Only `gpt-6-astra` and `gpt-5.4-mini` are recorded as `true`
  (every frozen run at `medium` reported reasoning tokens). A requested level is never dropped:
  known-unsupported fails, unknown is passed through with a warning. For Pi's generated
  `models.json`, the `reasoning` key is omitted when capability is unknown.

## Frozen treatments the plan does not (or cannot) reproduce

| Difference | Handling |
|---|---|
| Earliest Pi gpt-5.4-mini runs had **no network restriction** (representable) | Representable: `--network unrestricted` (plan warns). |
| `verification-v1` is **two agent passes** with a different second prompt | Not representable: a RunPlan describes one invocation. |
| Pi models.json was mounted at `/tmp` and `cp`'d into place | Plan names the final path only; the executor copies it in directly. |
| Frozen Pi/OpenCode metadata did not record network policy for Pi | Plans record it explicitly. |
| Pi custom-provider cost fields are zeros ("unpriced") | Reproduced as in the frozen config; such runs' cost metrics are not meaningful. |
| OpenCode reported non-zero cost for the self-hosted Qwen runs | Metric artefact of OpenCode's own pricing; not addressed here. |
