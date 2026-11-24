"""``ahl-experiment-v1``: the human-written experiment definition, and its normalized form.

The definition is a SET description: task, harness and model lists are order-insensitive, so the
normalized form sorts them. Two files that differ only in list order have the same hash and the
same matrix. Secrets never appear here: providers keep their credentials in environment
variables, referenced by NAME in the providers file.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent_harness_lab.runplan.models import DEFAULT_TIMEOUT_SECONDS, NetworkMode

EXPERIMENT_SCHEMA = "ahl-experiment-v1"

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,79}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REASONING = re.compile(r"^[a-z][a-z0-9_-]*$")
_HARNESS = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExperimentInfo(_Strict):
    name: str
    description: str = ""
    purpose: str = ""

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not _NAME.match(v):
            raise ValueError("name must be 1-80 characters: letters, digits, space, '.', '_', '-'")
        return v


class BenchmarkPin(_Strict):
    name: str = "swebench-multilingual"
    # Optional but strongly recommended: the checkout must be EXACTLY this commit.
    task_repository_commit: str | None = None

    @field_validator("task_repository_commit")
    @classmethod
    def _commit(cls, v: str | None) -> str | None:
        if v is not None:
            v = v.strip().lower()
            if not _COMMIT.match(v):
                raise ValueError("task_repository_commit must be a full 40-character commit hash")
        return v


class TaskSetFile(_Strict):
    """Tasks read from a task-set file (JSON/YAML), e.g. configs/challenge-set-v1.json."""

    file: str


class ModelEntry(_Strict):
    provider: str
    model: str
    reasoning: str | None = None

    @field_validator("provider", "model")
    @classmethod
    def _nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v

    @field_validator("reasoning")
    @classmethod
    def _reasoning(cls, v: str | None) -> str | None:
        if v is not None and not _REASONING.match(v):
            raise ValueError("reasoning must look like 'medium' (lowercase letters, digits, - _)")
        return v

    @property
    def ref(self) -> str:
        return f"{self.provider}/{self.model}"


class Execution(_Strict):
    timeout_seconds: int | None = Field(default=DEFAULT_TIMEOUT_SECONDS, ge=1)
    network: NetworkMode = NetworkMode.RESTRICTED_PROVIDER


class Evaluation(_Strict):
    enabled: bool = True


class ExperimentDefinition(_Strict):
    """The validated definition as written (task-set files not yet resolved)."""

    schema_version: Literal["ahl-experiment-v1"]
    experiment: ExperimentInfo
    benchmark: BenchmarkPin = BenchmarkPin()
    tasks: list[str] | TaskSetFile
    harnesses: list[str] = Field(min_length=1)
    models: list[ModelEntry] = Field(min_length=1)
    repetitions: int = Field(default=1, ge=1, le=100)
    execution: Execution = Execution()
    evaluation: Evaluation = Evaluation()

    @field_validator("harnesses")
    @classmethod
    def _harnesses(cls, v: list[str]) -> list[str]:
        for h in v:
            if not _HARNESS.match(h):
                raise ValueError(f"invalid harness name {h!r}")
        if len(set(v)) != len(v):
            raise ValueError("duplicate harness in harnesses")
        return v

    @model_validator(mode="after")
    def _unique(self) -> ExperimentDefinition:
        keys = [(m.provider, m.model, m.reasoning) for m in self.models]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate model entry (same provider, model and reasoning)")
        if isinstance(self.tasks, list):
            if not self.tasks:
                raise ValueError("tasks must not be empty")
            dupes = sorted({t for t in self.tasks if self.tasks.count(t) > 1})
            if dupes:
                raise ValueError(f"duplicate task(s): {', '.join(dupes)}")
        return self


class TaskSetRecord(_Strict):
    file: str  # as written in the experiment
    sha256: str  # of the file's bytes: the set is pinned by content


class NormalizedExperiment(_Strict):
    """What is hashed and stored: tasks resolved, every list sorted, defaults filled in."""

    schema_version: Literal["ahl-experiment-v1"] = EXPERIMENT_SCHEMA
    experiment: ExperimentInfo
    benchmark: BenchmarkPin
    tasks: list[str]
    task_set: TaskSetRecord | None = None
    harnesses: list[str]
    models: list[ModelEntry]
    repetitions: int
    execution: Execution
    evaluation: Evaluation

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def sanitize_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "experiment"
