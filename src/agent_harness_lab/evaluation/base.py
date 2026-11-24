"""Evaluation abstraction. Evaluation is benchmark-specific, never harness-specific: nothing in
this package receives a HarnessAdapter, and a run is described only by what is needed to score
its persisted ``model.patch``."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar, Protocol

from pydantic import BaseModel, ConfigDict

from agent_harness_lab.benchmarks.base import BenchmarkAdapter, RepositoryInfo
from agent_harness_lab.evaluation.models import EvaluationDocument, EvaluatorProvenance
from agent_harness_lab.execution.docker import CommandRunner, DockerClient
from agent_harness_lab.execution.preflight import CheckStatus, PreflightCheck


class EvaluationError(Exception):
    """A request that cannot be evaluated (reported without a traceback)."""


class RunNotFound(EvaluationError):
    pass


class UnsupportedBenchmark(EvaluationError):
    pass


class RunSubject(BaseModel):
    """What the evaluator needs to know about a run. Deliberately harness-free."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    run_dir: Path
    schema_name: str  # "ahl-run-v1" or the legacy metadata shape
    benchmark: str
    instance_id: str
    docker_image: str
    task_repository_commit: str | None  # commit recorded when the agent ran
    agent_image_id: str | None = None  # image id the agent actually ran in (run.json)
    patch_path: Path


@dataclass
class EvaluationContext:
    subject: RunSubject
    benchmark: BenchmarkAdapter  # ordinary task API only (existence/provenance)
    repository: RepositoryInfo
    python: Path | None  # interpreter of the environment that has the evaluator installed
    docker: DockerClient
    runner: CommandRunner
    environ: Mapping[str, str]
    output_dir: Path  # where evaluation artifacts and evaluation.json are written
    force: bool = False
    keep_evaluator_image: bool = False  # diagnostic: keep the evaluator-built image
    now: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))
    process_timeout_seconds: float | None = 7200  # whole evaluator process (builds + tests)


class EvaluationPreflight(BaseModel):
    checks: list[PreflightCheck] = []
    provisioning: list = []  # unused: the shared pre-flight renderer expects the field
    generated_files: list = []
    environment: EvaluatorProvenance | None = None
    evaluator_inputs_sha256: dict[str, str] = {}
    evaluator_build_inputs_sha256: dict[str, str] = {}
    image_id: str | None = None

    @property
    def ready(self) -> bool:
        return not any(c.status is CheckStatus.ERROR for c in self.checks)

    def add(self, name: str, status: CheckStatus, detail: str = "") -> None:
        self.checks.append(PreflightCheck(name=name, status=status, detail=detail))

    def model_dump_for_json(self) -> str:
        return json.dumps({**self.model_dump(mode="json"), "ready": self.ready}, indent=2)


class EvaluationAdapter(Protocol):
    name: ClassVar[str]
    benchmark: ClassVar[str]

    def preflight(self, ctx: EvaluationContext) -> EvaluationPreflight: ...

    def evaluate(
        self, ctx: EvaluationContext, preflight: EvaluationPreflight
    ) -> EvaluationDocument: ...

    def readiness(
        self,
        *,
        benchmark: BenchmarkAdapter,
        task_ids: Sequence[str],
        python: Path | None,
        runner: CommandRunner,
        environ: Mapping[str, str],
    ) -> list[PreflightCheck]: ...
