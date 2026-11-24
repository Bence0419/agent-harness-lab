"""SWE-bench Multilingual backend, reading an external swe-bench-multilingual-tasks checkout.

Only ``task.yaml`` and ``problem_statement.md`` are ever opened. Layout:
``<repo>/tasks/<instance_id>/{task.yaml, problem_statement.md, ...evaluator files...}``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, ClassVar

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from agent_harness_lab.benchmarks.base import (
    InvalidTaskRepository,
    MalformedTask,
    RepositoryInfo,
    Task,
    TaskListing,
    TaskNotFound,
    TaskProblem,
)
from agent_harness_lab.benchmarks.git import GitHead, GitProbe

TASKS_SUBDIR = "tasks"
TASK_FILE = "task.yaml"
STATEMENT_FILE = "problem_statement.md"

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class _TaskYaml(BaseModel):
    """Public fields of task.yaml. Unknown keys (e.g. eval_type, log_parser) are dropped."""

    model_config = ConfigDict(extra="ignore")

    instance_id: str
    repo: str
    base_commit: str
    image: str
    split: str | None = None
    version: str | None = None
    created_at: str | None = None

    @field_validator("instance_id", "repo", "base_commit", "image")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v

    @field_validator("version", "created_at", "split", mode="before")
    @classmethod
    def _scalar_to_str(cls, v: Any) -> Any:
        # YAML may turn unquoted versions/dates into numbers or dates.
        return v if v is None or isinstance(v, str) else str(v)


def looks_like_task_repo(path: Path) -> bool:
    return (path / TASKS_SUBDIR).is_dir()


class SWEBenchMultilingual:
    name: ClassVar[str] = "swebench-multilingual"
    container_workdir: ClassVar[str] = "/testbed"  # as used by the frozen runners

    def __init__(self, repo_path: Path, git: GitProbe | None = None) -> None:
        self._path = repo_path
        self._git = git or GitHead()
        self._info: RepositoryInfo | None = None

    # ---- repository -----------------------------------------------------
    def repository_info(self) -> RepositoryInfo:
        if self._info is None:
            if not self._path.is_dir():
                raise InvalidTaskRepository(f"task repository path does not exist: {self._path}")
            if not looks_like_task_repo(self._path):
                raise InvalidTaskRepository(
                    f"{self._path} does not look like a task repository "
                    f"(no '{TASKS_SUBDIR}/' directory)"
                )
            self._info = RepositoryInfo(
                benchmark=self.name,
                path=self._path.resolve(),
                commit=self._git.head_commit(self._path),
                dirty=self._git.is_dirty(self._path),
            )
        return self._info

    # ---- tasks ----------------------------------------------------------
    def list_tasks(self) -> TaskListing:
        info = self.repository_info()
        tasks: list[Task] = []
        problems: list[TaskProblem] = []
        root = info.path / TASKS_SUBDIR
        candidates = sorted(
            p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
        )
        for task_dir in candidates:
            try:
                tasks.append(self._load(task_dir, info))
            except MalformedTask as exc:
                problems.append(
                    TaskProblem(instance_id=task_dir.name, path=task_dir, reason=exc.reason)
                )
        return TaskListing(repository=info, tasks=tasks, problems=problems)

    def get_task(self, instance_id: str) -> Task:
        info = self.repository_info()
        if not _SAFE_ID.match(instance_id):
            raise TaskNotFound(f"invalid task id: {instance_id!r}")
        task_dir = info.path / TASKS_SUBDIR / instance_id
        if not task_dir.is_dir():
            raise TaskNotFound(f"task not found: {instance_id}")
        return self._load(task_dir, info)

    def _load(self, task_dir: Path, info: RepositoryInfo) -> Task:
        dir_id = task_dir.name
        meta = self._read_task_yaml(task_dir, dir_id)
        if meta.instance_id != dir_id:
            raise MalformedTask(
                dir_id, f"task.yaml instance_id {meta.instance_id!r} does not match directory name"
            )
        return Task(
            benchmark=self.name,
            instance_id=meta.instance_id,
            repo=meta.repo,
            base_commit=meta.base_commit,
            docker_image=meta.image,
            problem_statement=self._read_statement(task_dir, dir_id),
            task_directory=task_dir,
            task_repository_commit=info.commit,
            split=meta.split,
            version=meta.version,
            created_at=meta.created_at,
        )

    @staticmethod
    def _read_task_yaml(task_dir: Path, dir_id: str) -> _TaskYaml:
        path = task_dir / TASK_FILE
        if not path.is_file():
            raise MalformedTask(dir_id, f"missing {TASK_FILE}")
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError) as exc:
            raise MalformedTask(dir_id, f"cannot read {TASK_FILE}: {exc}") from exc
        except yaml.YAMLError as exc:
            raise MalformedTask(dir_id, f"invalid YAML in {TASK_FILE}: {exc}") from exc
        if not isinstance(data, dict):
            raise MalformedTask(dir_id, f"{TASK_FILE} is not a mapping")
        try:
            return _TaskYaml.model_validate(data)
        except ValidationError as exc:
            detail = "; ".join(
                f"{'.'.join(map(str, e['loc'])) or '?'}: {e['msg'].lower()}" for e in exc.errors()
            )
            raise MalformedTask(dir_id, f"invalid {TASK_FILE}: {detail}") from exc

    @staticmethod
    def _read_statement(task_dir: Path, dir_id: str) -> str:
        path = task_dir / STATEMENT_FILE
        if not path.is_file():
            raise MalformedTask(dir_id, f"missing {STATEMENT_FILE}")
        try:
            text = path.read_text(encoding="utf-8")  # strict: no silent replacement
        except UnicodeDecodeError as exc:
            raise MalformedTask(dir_id, f"{STATEMENT_FILE} is not valid UTF-8: {exc}") from exc
        except OSError as exc:
            raise MalformedTask(dir_id, f"cannot read {STATEMENT_FILE}: {exc}") from exc
        if not text.strip():
            raise MalformedTask(dir_id, f"{STATEMENT_FILE} is empty")
        return text
