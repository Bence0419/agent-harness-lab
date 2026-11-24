"""Canonical evaluation record (``evaluation.json``, schema ``ahl-evaluation-v1``).

Benchmark correctness is kept strictly apart from evaluation-infrastructure health:

* ``resolved = true``   : the official evaluator resolved the task
* ``resolved = false``  : the evaluator ran and the patch did NOT resolve it (unresolved/empty)
* ``resolved = null``   : no benchmark verdict (infrastructure failure, evaluator error,
                          incomplete or ambiguous output). Never counted as a FAIL.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from agent_harness_lab.artifacts.models import EvaluationState

EVALUATION_SCHEMA = "ahl-evaluation-v1"
EVALUATION_FILE = "evaluation.json"


class Classification(StrEnum):
    BENCHMARK_PASS = "benchmark_pass"
    BENCHMARK_FAIL = "benchmark_fail"
    EVALUATION_FAILURE = "evaluation_failure"  # infrastructure/evaluator problem: no verdict
    AMBIGUOUS = "ambiguous"  # evaluator output cannot be trusted either way: no verdict


# A patch the evaluator could not apply is the MODEL's failure, not the evaluator's.
_BENCHMARK_FAIL_STATES = (
    EvaluationState.UNRESOLVED,
    EvaluationState.EMPTY_PATCH,
    EvaluationState.PATCH_APPLY_FAILED,
)


def resolved_for(state: EvaluationState) -> bool | None:
    """The ONE mapping from evaluation state to a benchmark verdict (None = no verdict)."""
    if state is EvaluationState.RESOLVED:
        return True
    if state in _BENCHMARK_FAIL_STATES:
        return False
    return None


def classify(state: EvaluationState) -> Classification:
    if state is EvaluationState.RESOLVED:
        return Classification.BENCHMARK_PASS
    if state in _BENCHMARK_FAIL_STATES:
        return Classification.BENCHMARK_FAIL
    if state in (EvaluationState.AMBIGUOUS_FAILURE, EvaluationState.UNKNOWN):
        return Classification.AMBIGUOUS
    return Classification.EVALUATION_FAILURE


class EvaluatorImageBaseline(BaseModel):
    """Is the evaluator image really at the task's baseline? Environment provenance only.

    Anything other than ``verified`` means the verdict cannot be attributed to the model:
    the evaluation is classified as an infrastructure failure (no verdict).
    """

    verification_status: Literal["verified", "mismatch", "dirty", "unverified"]
    image_id: str | None  # the exact evaluator image inspected (never the shared tag)
    expected_base_commit: str
    actual_base_commit: str | None = None
    testbed_exists: bool | None = None
    is_git_repository: bool | None = None
    worktree_clean: bool | None = None
    reason: str | None = None  # machine-readable, e.g. evaluator_image_baseline_mismatch
    error: str | None = None


class EvaluatorProvenance(BaseModel):
    python_executable: str
    python_version: str
    evaluator_package: str
    evaluator_version: str | None
    cli: str


class EvaluatorCommand(BaseModel):
    argv: list[str]
    cwd: str
    eval_run_id: str
    model_name_or_path: str
    exit_code: int | None = None
    timed_out: bool = False


class ImageEvidence(BaseModel):
    """Which image was used when. The official evaluator REBUILDS the task image, so these
    are deliberately separate identities and are never conflated:

    * ``agent_image_id``      the image the agent ran in (from run.json; None if unknown)
    * ``image_id_before``     what the task tag pointed to when evaluation started
    * ``evaluator_image_id``  the image the evaluator built and ran its tests in
    * ``final_image_id``      what the tag points to afterwards (== before when restored)
    """

    reference: str
    agent_image_id: str | None = None
    image_id_before: str | None = None
    evaluator_image_id: str | None = None
    final_image_id: str | None = None
    rebuilt: bool = False
    restoration: Literal["not-needed", "restored", "failed", "skipped"] = "skipped"
    snapshot_ref: str | None = None
    snapshot_removed: bool = False
    evaluator_image_ref: str | None = None  # only when retention == "kept"
    retention: Literal[
        "not-applicable", "removed", "kept", "retained-for-recovery",
        "skipped-preexisting", "skipped-unsafe", "remove-failed",
    ] = "not-applicable"  # fmt: skip
    keep_requested: bool = False


class EvaluationDocument(BaseModel):
    schema_version: Literal["ahl-evaluation-v1"] = EVALUATION_SCHEMA
    run_id: str
    benchmark: str
    evaluator: str
    evaluator_version: str | None
    task: dict[str, str]  # instance_id (+ repo when known)

    started_at: str
    finished_at: str
    duration_seconds: float  # evaluation time only; NEVER merged into agent duration

    state: EvaluationState
    resolved: bool | None
    classification: Classification

    patch_sha256: str
    task_repository_commit: str | None  # commit of the task repo used for this evaluation
    run_task_repository_commit: str | None  # commit recorded by the run being evaluated
    image: ImageEvidence
    environment: EvaluatorProvenance
    command: EvaluatorCommand
    evaluator_inputs_sha256: dict[str, str] = Field(default_factory=dict)
    # build-defining PUBLIC files of the task (Dockerfile, task.yaml); supplements the commit
    evaluator_build_inputs_sha256: dict[str, str] = Field(default_factory=dict)
    evaluator_image_baseline: EvaluatorImageBaseline | None = None
    reason: str | None = None  # machine-readable cause when the state was decided by AHL
    official_result: dict = Field(default_factory=dict)  # this instance's entry in the report
    artifacts: dict[str, str] = Field(default_factory=dict)  # relative path -> sha256

    error: str | None = None
    warnings: list[str] = Field(default_factory=list)
    replaces: str | None = None  # archived previous evaluation, when --force replaced one
    interrupted: bool = False  # the evaluation was interrupted (Ctrl-C); there is no verdict
