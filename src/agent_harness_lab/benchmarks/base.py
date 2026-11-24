"""Benchmark abstraction and the normalized, evaluator-free Task model."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Protocol

from pydantic import BaseModel, ConfigDict


class BenchmarkError(Exception):
    """Base class for domain errors; the CLI reports these without a traceback."""


class TaskRepoNotConfigured(BenchmarkError):
    pass


class InvalidTaskRepository(BenchmarkError):
    pass


class TaskNotFound(BenchmarkError):
    pass


class MalformedTask(BenchmarkError):
    def __init__(self, instance_id: str, reason: str) -> None:
        super().__init__(f"{instance_id}: {reason}")
        self.instance_id = instance_id
        self.reason = reason


class CommitMismatch(BenchmarkError):
    def __init__(self, expected: str, actual: str | None) -> None:
        shown = actual or "unknown (not a readable git repository)"
        super().__init__(f"task repository commit mismatch: expected {expected}, found {shown}")
        self.expected = expected
        self.actual = actual


class Task(BaseModel):
    """What an agent run may know about a task.

    INTEGRITY: this model intentionally has no field for evaluator or gold
    material (gold patch, hints, test patch, tests list, eval script, log parser
    or eval type). The adapter never opens those files for normal task access;
    evaluation will get its own adapter and types. Note ``task_directory`` points
    at a directory that *contains* such files: it must never be mounted into or
    exposed to an agent.
    """

    model_config = ConfigDict(frozen=True)

    benchmark: str
    instance_id: str
    repo: str
    base_commit: str
    docker_image: str  # the image the task requires; whether it is present is not our concern
    problem_statement: str
    task_directory: Path
    task_repository_commit: str | None  # actual commit of the task repo, None if unreadable

    split: str | None = None
    version: str | None = None
    created_at: str | None = None


class RepositoryInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    benchmark: str
    path: Path
    commit: str | None
    dirty: bool | None = None  # uncommitted changes in the task repo (None = unknown)


class TaskProblem(BaseModel):
    """A task directory that could not be loaded. Always reported, never skipped silently."""

    model_config = ConfigDict(frozen=True)

    instance_id: str
    path: Path
    reason: str


class TaskListing(BaseModel):
    model_config = ConfigDict(frozen=True)

    repository: RepositoryInfo
    tasks: list[Task]
    problems: list[TaskProblem]


class BenchmarkAdapter(Protocol):
    name: ClassVar[str]
    container_workdir: ClassVar[str]  # repository location inside the task's container

    def repository_info(self) -> RepositoryInfo: ...

    def list_tasks(self) -> TaskListing: ...

    def get_task(self, instance_id: str) -> Task: ...


def verify_expected_commit(info: RepositoryInfo, expected: str | None) -> None:
    """Explicit pin check. ``expected=None`` means no pin. Never mutates the repository."""
    if expected is None:
        return
    if info.commit is None or info.commit.lower() != expected.strip().lower():
        raise CommitMismatch(expected.strip(), info.commit)
