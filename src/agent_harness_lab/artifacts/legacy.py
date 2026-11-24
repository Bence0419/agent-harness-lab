"""Read-only loader for historical (pre-``ahl``) run directories.

Nothing here writes to or rewrites run artifacts. Harness and format are taken
from structured fields in ``metadata.json``, never from directory names.

Historical SWE-bench ``metadata.json`` shapes:

``opencode-swebench``
    has ``opencodeVersion`` / ``opencodeExitCode`` (also ``variant``,
    ``networkPolicy``, ``squidImage``).
``pi-swebench-verification``
    Pi run with ``experiment`` + ``verificationPasses`` (solve + verify pass).
``pi-swebench-reference``
    Pi run with a ``reference`` identifier (``piExitCode``).
``pi-swebench-baseline``
    Pi run with neither ``reference`` nor ``experiment`` (earliest SWE-bench
    runs); the experiment identity is simply not recorded.

``evaluation.json`` (written by evaluate-swebench-run.sh) has one shape.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from agent_harness_lab.artifacts.ahl_run import load_ahl_run
from agent_harness_lab.artifacts.models import EvaluationState, RunRecord
from agent_harness_lab.artifacts.run_record import COMPAT_METADATA_SCHEMA, RUN_FILE


class LegacyFormatError(Exception):
    """The run directory is not a loadable run."""


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise LegacyFormatError(f"cannot read {path.name}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise LegacyFormatError(f"malformed {path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise LegacyFormatError(f"{path.name} is not a JSON object")
    return data


def detect_format(meta: dict[str, Any]) -> str:
    if "opencodeVersion" in meta or "opencodeExitCode" in meta:
        return "opencode-swebench"
    if "piExitCode" in meta or "piVersion" in meta:
        if "experiment" in meta and "verificationPasses" in meta:
            return "pi-swebench-verification"
        if "reference" in meta:
            return "pi-swebench-reference"
        return "pi-swebench-baseline"
    raise LegacyFormatError("unrecognized metadata.json shape")


def _str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _duration(start: datetime | None, end: datetime | None) -> int | None:
    if start is None or end is None:
        return None
    try:
        return int((end - start).total_seconds())
    except TypeError:  # naive vs aware
        return None


_STATE_FLAGS = (
    ("resolved", EvaluationState.RESOLVED),
    ("emptyPatch", EvaluationState.EMPTY_PATCH),
    ("infraFailure", EvaluationState.INFRA_FAILURE),
    ("ambiguousFailure", EvaluationState.AMBIGUOUS_FAILURE),
    ("error", EvaluationState.ERROR),
    ("unresolved", EvaluationState.UNRESOLVED),
    ("incomplete", EvaluationState.INCOMPLETE),
)

NEW_EVALUATION_SCHEMA = "ahl-evaluation-v1"


def verdict_for(state: EvaluationState) -> bool | None:
    """Benchmark verdict for a state: True/False, or None when there is NO verdict.

    Infrastructure failures, evaluator errors, incomplete and ambiguous results are not
    benchmark failures; they are excluded from correctness denominators.
    """
    if state is EvaluationState.RESOLVED:
        return True
    if state in (
        EvaluationState.UNRESOLVED,
        EvaluationState.EMPTY_PATCH,
        EvaluationState.PATCH_APPLY_FAILED,
    ):
        return False
    return None


def load_evaluation(run_dir: Path, run_id: str) -> tuple[EvaluationState, bool | None]:
    """Return (state, resolved). ``resolved`` is None unless the evaluation gives a verdict.

    Reads the canonical ``ahl-evaluation-v1`` schema (``schema_version``) for new runs and the
    historical flag-style shape (``schemaVersion``) for old ones; the two never overlap.
    """
    path = run_dir / "evaluation.json"
    if not path.is_file():
        return EvaluationState.NOT_EVALUATED, None
    try:
        data = _read_json_object(path)
    except LegacyFormatError:
        return EvaluationState.INVALID, None
    if data.get("schema_version") == NEW_EVALUATION_SCHEMA:
        return _load_new_evaluation(data, run_id)
    if data.get("generationRunId") not in (None, run_id):
        return EvaluationState.INVALID, None  # evaluation belongs to another run
    for flag, state in _STATE_FLAGS:
        if data.get(flag) is True:
            return state, verdict_for(state)
    return EvaluationState.UNKNOWN, None


def _load_new_evaluation(data: dict[str, Any], run_id: str) -> tuple[EvaluationState, bool | None]:
    if data.get("run_id") != run_id:
        return EvaluationState.INVALID, None
    try:
        state = EvaluationState(data["state"])
    except (KeyError, ValueError):
        return EvaluationState.INVALID, None
    return state, verdict_for(state)  # the verdict is derived from the state, never trusted blindly


def load_run(run_dir: Path) -> RunRecord:
    """Parse one run directory into a RunRecord.

    ``run.json`` (ahl-run-v1) is canonical and preferred whenever it exists. Directories
    without it are historical runs and go through the legacy ``metadata.json`` shapes.
    """
    if (run_dir / RUN_FILE).is_file():
        return load_ahl_run(run_dir)
    meta_path = run_dir / "metadata.json"
    if not meta_path.is_file():
        raise LegacyFormatError("no metadata.json (incomplete run?)")
    meta = _read_json_object(meta_path)
    if meta.get("schema") == COMPAT_METADATA_SCHEMA:
        raise LegacyFormatError(
            f"{COMPAT_METADATA_SCHEMA} compatibility metadata without run.json "
            "(run.json is the canonical record)"
        )
    fmt = detect_format(meta)

    run_id = _str(meta.get("runId"))
    task = _str(meta.get("instanceId"))
    if run_id is None or task is None:
        raise LegacyFormatError("metadata.json lacks runId or instanceId")

    if fmt == "opencode-swebench":
        harness, version = "OpenCode", _str(meta.get("opencodeVersion"))
        reasoning = _str(meta.get("variant"))
    else:
        harness, version = "Pi", _str(meta.get("piVersion"))
        reasoning = _str(meta.get("thinking"))

    reference = _str(meta.get("reference")) or _str(meta.get("experiment"))
    started = _parse_time(meta.get("startedAt"))
    finished = _parse_time(meta.get("finishedAt"))
    state, resolved = load_evaluation(run_dir, run_id)
    dirty = meta.get("harnessDirty")

    return RunRecord(
        run_id=run_id,
        task=task,
        harness=harness,
        harness_version=version,
        model=_str(meta.get("model")),
        provider=_str(meta.get("provider")),
        reasoning=reasoning,
        reference=reference,
        started_at=started,
        duration_seconds=_duration(started, finished),
        resolved=resolved,
        evaluation_state=state,
        harness_commit=_str(meta.get("harnessCommit")),
        harness_dirty=dirty if isinstance(dirty, bool) else None,
        task_repo_commit=_str(meta.get("taskRepoCommit")),
        base_commit=_str(meta.get("baseCommit")),
        docker_image=_str(meta.get("image")),
        network_policy=_str(meta.get("networkPolicy")),
        artifact_dir=run_dir,
        legacy_format=fmt,
    )
