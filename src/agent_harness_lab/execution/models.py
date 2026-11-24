"""Execution-layer data. No harness-specific knowledge lives here."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ExecutionState(StrEnum):
    PREPARED = "prepared"  # run directory + run.json created, nothing provisioned yet
    RUNNING = "running"  # the harness invocation has started
    COMPLETED = "completed"  # harness exited 0 and the patch was captured
    HARNESS_FAILED = "harness_failed"  # harness exited non-zero (evidence still captured)
    TIMED_OUT = "timed_out"  # RunPlan.timeout_seconds elapsed
    FAILED = "failed"  # infrastructure/stage failure; see failed_stage


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class Provision(_Frozen):
    """One thing placed into the task container: a bind mount or a file copy."""

    kind: Literal["bind-mount", "copy-in"]
    source: str  # host path
    destination: str  # container path
    read_only: bool
    purpose: str
    sha256: str | None = None  # of the file content, for generated files


class ImageRecord(_Frozen):
    reference: str  # as requested (historically a mutable ":latest" tag)
    image_id: str | None = None  # resolved content-addressed id
    repo_digests: list[str] = Field(default_factory=list)


class ContainerSpec(_Frozen):
    name: str
    image: str
    network: str | None
    env: dict[str, str]  # non-secret values only
    pass_env: list[str]  # secret NAMES: forwarded as `-e NAME` so values never enter argv
    mounts: list[Provision]  # bind mounts only
    labels: dict[str, str]


class ExecOutput(_Frozen):
    returncode: int | None
    timed_out: bool = False
    stdout: str = ""  # empty when output was redirected to a file
    stderr: str = ""


class NetworkRecord(_Frozen):
    mode: str
    network_name: str | None = None
    proxy_name: str | None = None
    squid_image: str | None = None
    squid_policy_sha256: str | None = None
    allowed_hosts: list[str] = Field(default_factory=list)


class CleanupRecord(_Frozen):
    status: Literal["not-needed", "ok", "failed"] = "not-needed"
    errors: list[str] = Field(default_factory=list)


class DockerError(Exception):
    """A Docker operation failed. The message never contains secret values."""


class ImageMissing(DockerError):
    pass


class StageError(Exception):
    """A named execution stage failed (recorded as ExecutionState.FAILED + failed_stage)."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage
        self.message = message


class ExecutionDisabled(Exception):
    """Real Pi/OpenCode execution is not enabled in this phase."""
