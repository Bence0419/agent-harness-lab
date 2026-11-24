"""Harness adapter abstraction. Adapters describe invocations; they never execute anything."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Protocol

from pydantic import BaseModel, ConfigDict, Field

from agent_harness_lab.execution.models import Provision
from agent_harness_lab.harnesses.provisioning import HostEnvironment
from agent_harness_lab.providers import ResolvedModel

# The agent prompt is the final argv element. The executor substitutes this exact token
# (as a whole argv element) with the prompt text, so no shell quoting is ever involved.
PROMPT_PLACEHOLDER = "<<AHL_PROMPT>>"


class HarnessError(Exception):
    """Invalid harness request. Messages never contain secret values."""


class GeneratedFile(BaseModel):
    """Configuration the harness needs inside the container, rendered by the adapter."""

    model_config = ConfigDict(frozen=True)

    container_path: str
    content: str
    description: str


class OutputExpectation(BaseModel):
    """Where the harness's output goes and how to parse it."""

    model_config = ConfigDict(frozen=True)

    trace_format: str  # key of the trace normalizer (RunRecord.harness value)
    stdout_file: str = "trace.jsonl"  # stdout is the JSON event stream
    stderr_file: str


class HarnessInvocation(BaseModel):
    model_config = ConfigDict(frozen=True)

    argv: list[str]  # argv[0] is the executable path inside the container
    prompt_placeholder: str = PROMPT_PLACEHOLDER
    env: dict[str, str] = Field(default_factory=dict)  # non-secret only
    secret_env: list[str] = Field(default_factory=list)  # variable NAMES, never values
    config_files: list[GeneratedFile] = Field(default_factory=list)
    output: OutputExpectation

    @property
    def executable(self) -> str:
        return self.argv[0]


class HarnessAdapter(Protocol):
    name: ClassVar[str]  # CLI name: "pi"
    display_name: ClassVar[str]  # matches RunRecord.harness: "Pi"
    executable: ClassVar[str]  # host executable to discover
    npm_package: ClassVar[str]  # key in package.json for the version pin

    def validate_reasoning(self, reasoning: str | None) -> None: ...

    def build_invocation(self, model: ResolvedModel) -> HarnessInvocation: ...

    def provisioning(
        self, executable_path: str, host: HostEnvironment, staged_files: dict[str, Path]
    ) -> list[Provision]:
        """What must be mounted/copied into the container for the invocation's in-container
        paths to exist. ``staged_files`` maps each generated file's container path to its
        host staging path."""
        ...
