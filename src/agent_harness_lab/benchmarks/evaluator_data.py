"""Evaluator-side task files: a SEPARATE API from the ordinary ``Task``.

This is the only place AHL touches the files the official evaluator consumes. The normal
benchmark adapter (``swebench_multilingual.py``) never imports this module, and ``Task`` has no
fields for any of it, so a harness/agent path cannot receive evaluator data by accident.

AHL reads only ``eval.sh``, ``test.patch`` and ``tests.json`` (hashed as evaluation provenance)
and the build-defining public files ``Dockerfile`` and ``task.yaml`` (hashed as build
provenance). ``gold.patch`` is checked for EXISTENCE only, because the official dataset loader
requires it; AHL never reads ``gold.patch`` or ``hints.md``.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from agent_harness_lab.artifacts.hashing import sha256_file

TASKS_SUBDIR = "tasks"
HASHED_FILES = ("eval.sh", "test.patch", "tests.json")  # read, to record provenance
# Build-defining public files used by the official image build (hashed as build provenance).
BUILD_FILES = ("Dockerfile", "task.yaml")
EXISTENCE_ONLY_FILES = ("gold.patch",)  # required by the loader; never read
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class EvaluatorFilesError(Exception):
    pass


class EvaluatorTaskFiles(BaseModel):
    model_config = ConfigDict(frozen=True)

    instance_id: str
    task_directory: Path
    missing: list[str]  # required evaluator files that are absent
    sha256: dict[str, str]  # of eval.sh / test.patch / tests.json (those present)
    build_sha256: dict[str, str]  # of Dockerfile / task.yaml (those present)

    @property
    def complete(self) -> bool:
        return not self.missing


def locate_evaluator_files(task_repo: Path, instance_id: str) -> EvaluatorTaskFiles:
    if not _SAFE_ID.match(instance_id):
        raise EvaluatorFilesError(f"invalid task id: {instance_id!r}")
    task_dir = task_repo / TASKS_SUBDIR / instance_id
    if not task_dir.is_dir():
        raise EvaluatorFilesError(f"task directory not found: {task_dir}")
    required = (*HASHED_FILES, *BUILD_FILES, *EXISTENCE_ONLY_FILES)
    missing = [name for name in required if not (task_dir / name).is_file()]
    hashes = {n: sha256_file(task_dir / n) for n in HASHED_FILES if (task_dir / n).is_file()}
    build = {n: sha256_file(task_dir / n) for n in BUILD_FILES if (task_dir / n).is_file()}
    return EvaluatorTaskFiles(
        instance_id=instance_id,
        task_directory=task_dir,
        missing=missing,
        sha256=hashes,
        build_sha256=build,
    )
