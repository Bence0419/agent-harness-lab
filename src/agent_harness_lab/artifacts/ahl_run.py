"""Loader for runs created by ``ahl`` (canonical ``run.json``, schema ``ahl-run-v1``)."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from agent_harness_lab.artifacts.models import RunRecord
from agent_harness_lab.artifacts.run_record import RUN_FILE, RUN_SCHEMA, RunDocument


def _error():
    from agent_harness_lab.artifacts.legacy import LegacyFormatError

    return LegacyFormatError


def load_ahl_run(run_dir: Path) -> RunRecord:
    from agent_harness_lab.artifacts.legacy import _parse_time, load_evaluation

    err = _error()
    path = run_dir / RUN_FILE
    try:
        doc = RunDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise err(f"cannot read {RUN_FILE}: {exc.strerror}") from exc
    except (ValidationError, ValueError) as exc:
        raise err(f"malformed or unsupported {RUN_FILE} (expected schema {RUN_SCHEMA})") from exc

    plan = doc.plan
    state, resolved = load_evaluation(run_dir, doc.run_id)
    return RunRecord(
        run_id=doc.run_id,
        task=plan.task.instance_id,
        harness=plan.harness.display_name,
        harness_version=plan.harness.version,
        model=plan.model.model,
        provider=plan.provider.id,
        reasoning=plan.model.reasoning,
        reference=None,
        started_at=_parse_time(doc.agent_started_at or doc.started_at),
        duration_seconds=None if doc.duration_seconds is None else int(doc.duration_seconds),
        resolved=resolved,
        evaluation_state=state,
        harness_commit=plan.ahl.commit,
        harness_dirty=plan.ahl.dirty,
        task_repo_commit=plan.task.task_repository_commit,
        base_commit=plan.task.base_commit,
        docker_image=plan.task.docker_image,
        network_policy=plan.network.mode.value,
        artifact_dir=run_dir,
        legacy_format=doc.schema_version,
        execution_state=doc.state.value,
        purpose=plan.purpose,
    )
