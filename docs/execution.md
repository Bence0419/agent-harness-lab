# Execution layer

The generic executor runs one `RunPlan` in an isolated task container and persists the evidence.
It knows Docker, the workspace contract and the artifact layout, and nothing about how Pi or
OpenCode are invoked (that lives in the plan's `HarnessInvocation`). Real harnesses require the
pre-release gate `AHL_ENABLE_REAL_EXECUTION=1` (see `docs/real-execution.md`), which the
executor re-checks itself. The layer is tested with a deterministic mock harness
(`tests/mock_harness.py`, never in the public registry) and a fake Docker client.

## Flow and stages

RunPlan -> new run dir -> prompt/config staging -> image inspect (no pull/build) -> provisioning
-> network -> container -> workspace reset -> harness exec -> patch capture -> cleanup ->
`run.json` + `metadata.json`. A failure becomes `state=failed` with `failed_stage` in
`environment | prompt | artifacts | image | provisioning | network | container | workspace |
harness | patch | internal | interrupted`. States: `prepared, running, completed,
harness_failed, timed_out, failed`. Cleanup always runs; its failure is recorded in
`cleanup` and never replaces the original failure. `run.json` is written atomically at the
start, when the harness starts, and at the end.

## Provisioning contract (from the frozen runners)

| Harness | Provision |
|---|---|
| Pi | host Node root -> `/opt/node24` (ro); project `node_modules` -> `/opt/pi-node_modules` (ro); generated `models.json` copied to `/root/.pi/agent/models.json` (the frozen runners mounted it at `/tmp` and `cp`'d it) |
| OpenCode | symlink-resolved binary -> `/usr/local/bin/opencode` (ro); generated config -> `/etc/opencode/opencode.json` (ro bind mount) |

The task directory is never mounted or copied; the plan has no field for it. Secrets are
forwarded as `docker create -e NAME` (value taken from the docker CLI's environment), so values
never enter argv, `run.json` or logs.

## Network

* `unrestricted`: default Docker bridge.
* `restricted-provider`: `--internal` network + pinned Squid (`ubuntu/squid@sha256:6a09...`,
  alias `squid`, also attached to `bridge`), policy rendered from the provider host, its sha256
  recorded; `proxy.log` collected before removal.

## `run.json` (`ahl-run-v1`)

`schema_version, run_id, state, failed_stage, error, started_at, finished_at, agent_started_at,
agent_finished_at, duration_seconds, exit_code, plan (full secret-free RunPlan), image
{reference, image_id, repo_digests}, container_name, container_id, provisioning[], network,
hashes {prompt_sha256, patch_sha256, trace_sha256, generated_config_sha256, artifacts}, cleanup,
warnings`. `metadata.json` (`schema: ahl-metadata-v1`) is a derived compatibility file, not one
of the historical shapes; the result loader prefers `run.json` and rejects a bare compat file.

## Known gaps / limits

* On timeout the host-side `docker exec` client gets SIGTERM, then SIGKILL after 30 s. Because
  that does not stop the in-container process, only this run's container is killed and then
  restarted (its filesystem survives) so the patch is captured from a quiescent workspace.
* Pulling/building task images is not implemented (missing image => `failed_stage=image`).
