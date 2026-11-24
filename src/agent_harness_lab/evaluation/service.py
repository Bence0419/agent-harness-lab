"""Resolve a run, run pre-flight, evaluate, and persist ``evaluation.json`` safely."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from agent_harness_lab.artifacts.legacy import LegacyFormatError, detect_format
from agent_harness_lab.artifacts.run_record import RUN_FILE, RunDocument
from agent_harness_lab.artifacts.writer import write_json_atomic
from agent_harness_lab.evaluation.base import (
    EvaluationAdapter,
    EvaluationContext,
    EvaluationError,
    EvaluationPreflight,
    RunNotFound,
    RunSubject,
)
from agent_harness_lab.evaluation.models import EVALUATION_FILE, EvaluationDocument
from agent_harness_lab.execution.preflight import CheckStatus

SWEBENCH_SUBDIR = "swebench"
LEGACY_BENCHMARK = "swebench-multilingual"  # historical runs all came from this benchmark


def resolve_run_dir(raw_dir: Path, ref: str) -> Path:
    """``ref`` is a run directory path or a run id looked up in the result store."""
    as_path = Path(ref)
    if as_path.is_dir() and (
        (as_path / RUN_FILE).is_file() or (as_path / "metadata.json").is_file()
    ):
        return as_path
    matches = sorted((raw_dir / SWEBENCH_SUBDIR).glob(f"*/{ref}")) if raw_dir.is_dir() else []
    matches = [m for m in matches if m.is_dir()]
    if not matches:
        raise RunNotFound(f"run not found: {ref} (looked in {raw_dir / SWEBENCH_SUBDIR})")
    return matches[0]


def load_subject(run_dir: Path) -> RunSubject:
    """Describe a run for evaluation. Canonical run.json is preferred; historical
    metadata.json runs are accepted when they carry enough information."""
    patch = run_dir / "model.patch"
    if (run_dir / RUN_FILE).is_file():
        try:
            doc = RunDocument.model_validate_json((run_dir / RUN_FILE).read_text(encoding="utf-8"))
        except (ValidationError, ValueError, OSError) as exc:
            raise EvaluationError(f"unsupported or malformed {RUN_FILE} in {run_dir.name}") from exc
        subject = RunSubject(
            run_id=doc.run_id,
            run_dir=run_dir,
            schema_name=doc.schema_version,
            benchmark=doc.plan.benchmark,
            instance_id=doc.plan.task.instance_id,
            docker_image=doc.plan.task.docker_image,
            task_repository_commit=doc.plan.task.task_repository_commit,
            agent_image_id=doc.image.image_id if doc.image else None,
            patch_path=patch,
        )
    else:
        try:
            meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
            fmt = detect_format(meta)
            subject = RunSubject(
                run_id=meta["runId"],
                run_dir=run_dir,
                schema_name=fmt,
                benchmark=LEGACY_BENCHMARK,
                instance_id=meta["instanceId"],
                docker_image=meta["image"],
                task_repository_commit=meta.get("taskRepoCommit"),
                patch_path=patch,
            )
        except (OSError, ValueError, KeyError, LegacyFormatError) as exc:
            raise EvaluationError(f"unsupported run schema in {run_dir.name}: {exc}") from exc
    if not patch.is_file():
        raise EvaluationError(f"run {subject.run_id} has no model.patch to evaluate")
    return subject


def preflight(adapter: EvaluationAdapter, ctx: EvaluationContext) -> EvaluationPreflight:
    """Adapter pre-flight plus the benchmark-independent overwrite check. Read-only."""
    pre = adapter.preflight(ctx)
    existing = ctx.output_dir / EVALUATION_FILE
    if existing.exists():
        if ctx.force:
            pre.add(
                "existing evaluation",
                CheckStatus.WARNING,
                "will be archived (--force), not deleted",
            )
        else:
            pre.add(
                "existing evaluation",
                CheckStatus.ERROR,
                f"{existing} already exists; use --force to re-evaluate",
            )
    else:
        pre.add("existing evaluation", CheckStatus.OK, "none")
    return pre


def run_evaluation(
    adapter: EvaluationAdapter, ctx: EvaluationContext
) -> tuple[EvaluationPreflight, EvaluationDocument | None]:
    """Evaluate if pre-flight passes. Never overwrites evidence: --force archives first."""
    pre = preflight(adapter, ctx)
    if not pre.ready:
        return pre, None

    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    replaces: str | None = None
    existing = ctx.output_dir / EVALUATION_FILE
    stamp = ctx.now().strftime("%Y%m%dT%H%M%SZ")
    if existing.exists():  # only reachable with --force
        archived = ctx.output_dir / f"evaluation.superseded-{stamp}.json"
        existing.rename(archived)
        replaces = archived.name
    old_work = ctx.output_dir / "evaluator"
    if ctx.force and old_work.exists():
        # also covers an evaluation killed before it wrote evaluation.json: its partial working
        # directory must not block a retry (experiment resume relies on this)
        old_work.rename(ctx.output_dir / f"evaluator.superseded-{stamp}")
    if (ctx.output_dir / "evaluator").exists():
        raise EvaluationError(
            "an evaluator working directory already exists; use --force to archive it"
        )

    doc = adapter.evaluate(ctx, pre)
    doc = doc.model_copy(update={"replaces": replaces})
    write_json_atomic(existing, doc.model_dump(mode="json"))
    if doc.interrupted:  # evidence is saved (no verdict); now honour the interruption
        raise KeyboardInterrupt
    return pre, doc
