"""The canonical run record (``run.json``, schema ``ahl-run-v1``) and the compatibility
``metadata.json`` derived from it.

run.json is the source of truth for runs created by ``ahl``. The legacy ``metadata.json``
shapes of the research Bash runners are only ever read by the legacy loader. The compatibility
file below has its own explicit schema id and is NOT one of those historical shapes.

Secrets: only environment-variable NAMES appear (inside the embedded RunPlan).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from agent_harness_lab.execution.models import (
    CleanupRecord,
    ExecutionState,
    ImageRecord,
    NetworkRecord,
    Provision,
)
from agent_harness_lab.runplan.models import RunPlan

RUN_SCHEMA = "ahl-run-v1"
COMPAT_METADATA_SCHEMA = "ahl-metadata-v1"
RUN_FILE = "run.json"
METADATA_FILE = "metadata.json"


class Hashes(BaseModel):
    prompt_sha256: str
    patch_sha256: str | None = None
    trace_sha256: str | None = None
    generated_config_sha256: dict[str, str] = Field(default_factory=dict)  # by container path
    artifacts: dict[str, str] = Field(default_factory=dict)  # file name -> sha256 (finalized)


class RunDocument(BaseModel):
    schema_version: Literal["ahl-run-v1"] = RUN_SCHEMA
    run_id: str
    state: ExecutionState
    failed_stage: str | None = None
    error: str | None = None

    started_at: str  # UTC ISO-8601
    finished_at: str | None = None
    agent_started_at: str | None = None
    agent_finished_at: str | None = None
    duration_seconds: float | None = None  # agent execution time
    exit_code: int | None = None
    # After an execution timeout: was this run's container killed? (None = no timeout occurred)
    timeout_action: Literal["killed", "kill-failed"] | None = None

    # Full resolved, secret-free plan: task, harness, model, timeout, network, AHL provenance.
    plan: RunPlan
    image: ImageRecord | None = None
    container_name: str | None = None
    container_id: str | None = None
    provisioning: list[Provision] = Field(default_factory=list)
    network: NetworkRecord | None = None
    hashes: Hashes
    cleanup: CleanupRecord = Field(default_factory=CleanupRecord)
    warnings: list[str] = Field(default_factory=list)


def compat_metadata(doc: RunDocument) -> dict[str, Any]:
    """Compatibility ``metadata.json`` for tooling written against metadata files."""
    plan = doc.plan
    return {
        "schema": COMPAT_METADATA_SCHEMA,
        "note": "derived from run.json, which is canonical",
        "runId": doc.run_id,
        "instanceId": plan.task.instance_id,
        "purpose": plan.purpose,
        "state": doc.state.value,
        "exitCode": doc.exit_code,
        "harness": plan.harness.display_name,
        "harnessVersion": plan.harness.version,
        "provider": plan.provider.id,
        "model": plan.model.model,
        "reasoning": plan.model.reasoning,
        "timeoutSeconds": plan.timeout_seconds,
        "networkMode": plan.network.mode.value,
        "taskRepoCommit": plan.task.task_repository_commit,
        "baseCommit": plan.task.base_commit,
        "image": plan.task.docker_image,
        "imageId": doc.image.image_id if doc.image else None,
        "ahlCommit": plan.ahl.commit,
        "ahlDirty": plan.ahl.dirty,
        "startedAt": doc.agent_started_at or doc.started_at,
        "finishedAt": doc.agent_finished_at or doc.finished_at,
    }
