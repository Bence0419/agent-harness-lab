"""RunPlan: the fully resolved, secret-free description of one run.

It is NOT run.json. It is what an executor (Phase 6) will carry out without reinterpreting
user intent, and the conceptual basis of the future canonical run record. Secrets are
represented only by environment-variable NAMES.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_harness_lab.harnesses.base import HarnessInvocation
from agent_harness_lab.providers import ModelSpec

RUNPLAN_SCHEMA = "ahl-runplan-v0"

# Why a run exists. "canary": a manual, pre-experiment sanity run. It never belongs to a named
# experiment, and comparisons exclude canaries by default. "experiment": one matrix cell of an
# ``ahl benchmark`` experiment; it carries an ExperimentRef.
Purpose = Literal["canary", "experiment"]
DEFAULT_TIMEOUT_SECONDS = 3600  # the frozen research default (AGENT_TIMEOUT_SECONDS)


class NetworkMode(StrEnum):
    RESTRICTED_PROVIDER = "restricted-provider"  # egress only to the model provider's host
    UNRESTRICTED = "unrestricted"  # no egress restriction (e.g. the earliest Pi baseline runs)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class TaskRef(_Frozen):
    instance_id: str
    repo: str
    base_commit: str
    docker_image: str
    task_repository_commit: str | None
    task_repository_dirty: bool | None
    split: str | None = None
    version: str | None = None


class PromptRef(_Frozen):
    preamble: str
    delivered_sha256: str  # hash of the exact prompt text passed to the agent
    delivered_chars: int
    delivery: Literal["final-argument"] = "final-argument"


class HarnessRef(_Frozen):
    name: str
    display_name: str
    executable_path: str  # host path of the discovered executable
    discovery_source: Literal["PATH", "project-local"]
    version: str
    expected_version: str | None  # the repository's package.json pin, if available


class ProviderRef(_Frozen):
    id: str
    type: Literal["openai", "openai-compatible"]
    display_name: str | None
    base_url: str | None  # resolved endpoint for openai-compatible providers
    base_url_env: str | None
    api_key_env: str  # a NAME, never a value


class NetworkPlan(_Frozen):
    mode: NetworkMode
    allowed_hosts: list[str]  # empty when unrestricted


class ModelCapabilities(_Frozen):
    """What is KNOWN about the model. None = unknown (never guessed)."""

    reasoning_supported: bool | None


class AhlRef(_Frozen):
    commit: str | None  # None outside a git source checkout (e.g. an installed wheel)
    dirty: bool | None
    version: str | None = None  # the installed AHL version (always known)


class ExperimentRef(_Frozen):
    """Where an experiment run belongs. Identifiers and hashes only: no secrets."""

    experiment_id: str
    definition_sha256: str  # of the normalized experiment definition
    matrix_cell_id: str
    repetition: int
    attempt: int = 1  # one matrix cell may have several attempts; each is a separate run


class RunPlan(_Frozen):
    schema_version: Literal["ahl-runplan-v0"] = RUNPLAN_SCHEMA
    purpose: Purpose = "canary"
    benchmark: str
    task: TaskRef
    prompt: PromptRef
    workdir: str  # repository location inside the task container
    harness: HarnessRef
    invocation: HarnessInvocation
    provider: ProviderRef
    model: ModelSpec
    model_capabilities: ModelCapabilities
    # A positive int is an execution time limit; None means NO execution timeout (as in some
    # frozen runners). New CLI runs default to DEFAULT_TIMEOUT_SECONDS.
    timeout_seconds: int | None
    network: NetworkPlan
    required_secret_env: list[str]
    ahl: AhlRef
    experiment: ExperimentRef | None = None  # set exactly when purpose == "experiment"

    @model_validator(mode="after")
    def _purpose_matches_experiment(self) -> RunPlan:
        if (self.purpose == "experiment") != (self.experiment is not None):
            raise ValueError("purpose 'experiment' requires an experiment reference, and only it")
        return self


class PlanResult(_Frozen):
    plan: RunPlan
    warnings: list[str] = Field(default_factory=list)
    secret_env_present: dict[str, bool] = Field(default_factory=dict)  # names only
