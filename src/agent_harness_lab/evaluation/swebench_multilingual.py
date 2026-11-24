"""Official SWE-bench Multilingual evaluation, reproducing scripts/evaluate-swebench-run.sh.

The frozen script ran ``swebench eval multilingual -p <predictions> -i <instance> --run-id <id>
--task-repo <task repo> -j 1 -t 1800`` and read the official report's id lists. This adapter
does the same, but in a private working directory inside the run's output directory (via
``--report-dir`` and the process cwd) so the evaluator's report and logs stay with the run.
It invents no PASS/FAIL heuristic: the verdict is the official report's classification.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

from agent_harness_lab.artifacts.hashing import sha256_file
from agent_harness_lab.artifacts.models import EvaluationState
from agent_harness_lab.artifacts.writer import write_text_atomic
from agent_harness_lab.benchmarks.base import BenchmarkAdapter
from agent_harness_lab.benchmarks.evaluator_data import EvaluatorFilesError, locate_evaluator_files
from agent_harness_lab.evaluation.base import (
    EvaluationContext,
    EvaluationError,
    EvaluationPreflight,
)
from agent_harness_lab.evaluation.baseline import (
    REASON_UNVERIFIED,
    verify_baseline,
)
from agent_harness_lab.evaluation.environment import (
    EvaluatorEnvironmentError,
    cli_path,
    detect_environment,
    scrubbed_env,
)
from agent_harness_lab.evaluation.image_guard import ImageGuard
from agent_harness_lab.evaluation.models import (
    EvaluationDocument,
    EvaluatorCommand,
    EvaluatorImageBaseline,
    EvaluatorProvenance,
    ImageEvidence,
    classify,
    resolved_for,
)
from agent_harness_lab.execution.docker import CommandRunner
from agent_harness_lab.execution.models import DockerError
from agent_harness_lab.execution.preflight import CheckStatus, PreflightCheck

DATASET = "multilingual"
WORKERS = 1
INSTANCE_TIMEOUT_SECONDS = 1800  # frozen script: -t 1800
WORK_DIRNAME = "evaluator"

# swebench 5.0.2 report semantics (verified against its source and by experiments):
# * exactly one PRIMARY list holds an instance: resolved / unresolved / empty_patch / error /
#   incomplete.
# * ``infra_failure_ids`` and ``ambiguous_failure_ids`` are advisory TAGS over unresolved|error
#   (signature matches in the logs): the instance also stays in its primary list.
# * ``error_ids`` means "no valid per-instance report.json was produced". The report alone does
#   not say why; the per-instance log does (see markers below).
_PRIMARY: tuple[tuple[EvaluationState, str], ...] = (
    (EvaluationState.RESOLVED, "resolved_ids"),
    (EvaluationState.UNRESOLVED, "unresolved_ids"),
    (EvaluationState.EMPTY_PATCH, "empty_patch_ids"),
    (EvaluationState.ERROR, "error_ids"),
    (EvaluationState.INCOMPLETE, "incomplete_ids"),
)

# Official strings written to run_instance.log (swebench.harness.constants / run_evaluation).
APPLY_PATCH_FAIL_MARKER = ">>>>> Patch Apply Failed"
TEST_TIMEOUT_MARKERS = ("Test timed out after", "Timeout error:")


def classify_report(
    report: dict[str, Any] | None,
    instance_id: str,
    *,
    exit_code: int | None,
    timed_out: bool,
    instance_report: dict[str, Any] | None = None,
    instance_log: str | None = None,
) -> tuple[EvaluationState, dict[str, Any], list[str], str | None]:
    """(state, this instance's official result, warnings, error). Pure and fully testable.

    The official report decides, refined only by official evidence (its tags, per-instance
    report.json and the log marker for patch-application failure):

    * resolved / unresolved / empty_patch  -> as reported;
    * unresolved + environment-tier tag    -> infra_failure (the environment broke, not the model);
    * unresolved + ambiguous-tier tag      -> ambiguous_failure (cannot tell);
    * error + "Patch Apply Failed" in log  -> patch_apply_failed: a MODEL failure, kept in the
      denominator (the evaluator calls it "error" but nothing is wrong with the evaluator);
    * error + environment-tier tag         -> infra_failure;
    * error + timeout / ambiguous tag      -> ambiguous_failure (slow model code or slow env);
    * any other error                      -> error (evaluator problem, cause unknown: no verdict);
    * no report / timeout of the evaluator -> infra_failure.
    Success is never inferred from anything ambiguous.
    """
    warnings: list[str] = []
    if timed_out:
        return EvaluationState.INFRA_FAILURE, {}, warnings, "evaluator process timed out"
    if report is None:
        return (
            EvaluationState.INFRA_FAILURE,
            {},
            warnings,
            f"evaluator produced no readable report (exit code {exit_code})",
        )

    hits = [s for s, key in _PRIMARY if instance_id in (report.get(key) or [])]
    infra_tag = instance_id in (report.get("infra_failure_ids") or [])
    ambiguous_tag = instance_id in (report.get("ambiguous_failure_ids") or [])
    log = instance_log or ""
    markers = [m for m in (APPLY_PATCH_FAIL_MARKER, *TEST_TIMEOUT_MARKERS) if m in log]
    official: dict[str, Any] = {
        "buckets": [s.value for s in hits],
        "infra_failure_tag": infra_tag,
        "ambiguous_failure_tag": ambiguous_tag,
        "completed": instance_id in (report.get("completed_ids") or []),
        "submitted": instance_id in (report.get("submitted_ids") or []),
        "report_schema_version": report.get("schema_version"),
        "instance_report": instance_report,
        "log_markers": markers,
    }
    reasons = report.get("failure_reasons")
    reason = reasons.get(instance_id) if isinstance(reasons, dict) else None
    if reason is not None:
        official["failure_reason"] = reason

    if not hits:
        return (
            EvaluationState.UNKNOWN,
            official,
            warnings,
            "instance is absent from every primary result list of the official report",
        )
    if len(hits) > 1:
        return (
            EvaluationState.AMBIGUOUS_FAILURE,
            official,
            warnings,
            f"instance appears in several primary result lists: {[h.value for h in hits]}",
        )
    primary = hits[0]
    detail = f" ({reason})" if reason else ""

    if primary is EvaluationState.RESOLVED:
        if not official["completed"] or exit_code != 0:
            return (
                EvaluationState.AMBIGUOUS_FAILURE,
                official,
                warnings,
                "report says resolved but the run is not cleanly completed "
                f"(completed={official['completed']}, exit code {exit_code}); not trusting it",
            )
        return primary, official, warnings, None

    if primary in (EvaluationState.EMPTY_PATCH, EvaluationState.INCOMPLETE):
        return primary, official, warnings, None

    if primary is EvaluationState.UNRESOLVED:
        if infra_tag:
            return (
                EvaluationState.INFRA_FAILURE,
                official,
                warnings,
                f"unresolved, but the official infrastructure signature fired{detail}",
            )
        if ambiguous_tag:
            return (
                EvaluationState.AMBIGUOUS_FAILURE,
                official,
                warnings,
                f"unresolved, but the official ambiguous signature fired{detail}; cannot "
                "tell a bad patch from a broken environment",
            )
        if exit_code not in (0, None):
            warnings.append(f"evaluator exited with code {exit_code} but reported unresolved")
        return primary, official, warnings, None

    # primary is ERROR: no valid per-instance report; the log says why.
    if APPLY_PATCH_FAIL_MARKER in markers:
        return (
            EvaluationState.PATCH_APPLY_FAILED,
            official,
            warnings,
            "the evaluator could not apply the model's patch (a model failure)",
        )
    if infra_tag:
        return (
            EvaluationState.INFRA_FAILURE,
            official,
            warnings,
            f"evaluator error with an infrastructure signature{detail}",
        )
    if ambiguous_tag or any(m in markers for m in TEST_TIMEOUT_MARKERS):
        return (
            EvaluationState.AMBIGUOUS_FAILURE,
            official,
            warnings,
            "evaluator error that may be the patch or the environment"
            f"{detail or ' (test timeout)'}",
        )
    return (
        EvaluationState.ERROR,
        official,
        warnings,
        "evaluator error with no recognised cause (no valid report, no patch-apply failure, "
        "no timeout, no signature): no benchmark verdict",
    )


# States whose attribution to the MODEL depends on the evaluator image's baseline being right.
# (empty_patch does not: with no patch there is nothing to apply.)
_MODEL_ATTRIBUTABLE = (
    EvaluationState.RESOLVED,
    EvaluationState.UNRESOLVED,
    EvaluationState.PATCH_APPLY_FAILED,
)


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "-", text)


class SWEBenchMultilingualEvaluator:
    name: ClassVar[str] = "swebench-multilingual"
    benchmark: ClassVar[str] = "swebench-multilingual"

    # ---- pre-flight (read-only) --------------------------------------------------------------
    def preflight(self, ctx: EvaluationContext) -> EvaluationPreflight:
        pre = EvaluationPreflight()
        ok, warn, err = CheckStatus.OK, CheckStatus.WARNING, CheckStatus.ERROR
        s = ctx.subject
        pre.add("run", ok, f"{s.run_id} ({s.schema_name}), benchmark {s.benchmark}")

        # model.patch
        if not s.patch_path.is_file():
            pre.add("model.patch", err, f"missing: {s.patch_path}")
        else:
            try:
                s.patch_path.read_text(encoding="utf-8")
                pre.add("model.patch", ok, f"{s.patch_path.stat().st_size} bytes, UTF-8")
            except UnicodeDecodeError:
                pre.add("model.patch", err, "not valid UTF-8; cannot form a predictions file")

        # task repository + task
        repo = ctx.repository
        if repo.commit is None:
            pre.add("task repository", err, "commit is unknown (not a readable git checkout)")
        else:
            pre.add("task repository", ok, f"{repo.path} @ {repo.commit}")
            if repo.dirty:
                pre.add("task repository state", warn, "uncommitted changes (DIRTY)")
            if s.task_repository_commit and s.task_repository_commit != repo.commit:
                pre.add(
                    "task repository version",
                    warn,
                    f"differs from the commit recorded by the run ({s.task_repository_commit})",
                )
        try:
            ctx.benchmark.get_task(s.instance_id)
            pre.add("task", ok, s.instance_id)
        except Exception as exc:  # BenchmarkError subclasses: unknown/malformed task
            pre.add("task", err, str(exc))

        # evaluator-side task files (existence + provenance hashes; contents never interpreted)
        try:
            files = locate_evaluator_files(repo.path, s.instance_id)
            if files.complete:
                pre.add(
                    "evaluator task files",
                    ok,
                    "eval.sh, test.patch, tests.json, Dockerfile, gold.patch",
                )
            else:
                pre.add("evaluator task files", err, f"missing: {', '.join(files.missing)}")
            pre.evaluator_inputs_sha256 = files.sha256
            pre.evaluator_build_inputs_sha256 = files.build_sha256
        except EvaluatorFilesError as exc:
            pre.add("evaluator task files", err, str(exc))

        # evaluator environment
        self._check_environment(ctx, pre)

        # Docker image (read-only inspection)
        try:
            rec = ctx.docker.image_inspect(s.docker_image)
            pre.image_id = rec.image_id
            pre.add("task image", ok, f"{s.docker_image} ({(rec.image_id or '')[:19]})")
        except DockerError as exc:
            pre.add("task image", err, str(exc))
        return pre

    def readiness(
        self,
        *,
        benchmark: BenchmarkAdapter,
        task_ids: Sequence[str],
        python: Path | None,
        runner: CommandRunner,
        environ: Mapping[str, str],
    ) -> list[PreflightCheck]:
        """Can this evaluator score runs of these tasks? Used BEFORE any run exists (experiment
        pre-flight). Read-only: existence checks and the evaluator-environment probe only."""
        pre = EvaluationPreflight()
        self._check_environment_for(python, runner, environ, pre)
        path = benchmark.repository_info().path
        missing: dict[str, list[str]] = {}
        for task_id in task_ids:
            try:
                files = locate_evaluator_files(path, task_id)
            except EvaluatorFilesError as exc:
                missing[task_id] = [str(exc)]
                continue
            if not files.complete:
                missing[task_id] = list(files.missing)
        if missing:
            detail = "; ".join(f"{t}: {', '.join(m)}" for t, m in sorted(missing.items()))
            pre.add("evaluator task files", CheckStatus.ERROR, detail)
        else:
            pre.add("evaluator task files", CheckStatus.OK, f"complete for {len(task_ids)} task(s)")
        return pre.checks

    def _check_environment(self, ctx: EvaluationContext, pre: EvaluationPreflight) -> None:
        self._check_environment_for(ctx.python, ctx.runner, ctx.environ, pre)

    def _check_environment_for(
        self,
        python: Path | None,
        runner: CommandRunner,
        environ: Mapping[str, str],
        pre: EvaluationPreflight,
    ) -> None:
        ok, err = CheckStatus.OK, CheckStatus.ERROR
        if python is None:
            pre.add(
                "evaluator python",
                err,
                "not configured: pass --swebench-python or set AHL_SWEBENCH_PYTHON",
            )
            return
        if not python.is_file():
            pre.add("evaluator python", err, f"not found: {python}")
            return
        pre.add("evaluator python", ok, str(python))
        if not cli_path(python).is_file():
            pre.add(
                "evaluator CLI", err, f"missing next to the interpreter: {cli_path(python)}"
            )
        else:
            pre.add("evaluator CLI", ok, str(cli_path(python)))
        try:
            env = detect_environment(python, runner, environ)
        except EvaluatorEnvironmentError as exc:
            pre.add("evaluator package", err, str(exc))
            return
        pre.environment = env
        pre.add(
            "evaluator package",
            ok,
            f"swebench {env.evaluator_version} (Python {env.python_version})",
        )

    # ---- evaluation ----
    def evaluate(self, ctx: EvaluationContext, pre: EvaluationPreflight) -> EvaluationDocument:
        assert pre.environment is not None and ctx.python is not None
        s = ctx.subject
        started = ctx.now()
        work = ctx.output_dir / WORK_DIRNAME
        work.mkdir(parents=True, exist_ok=False)
        stamp = started.strftime("%Y%m%dT%H%M%SZ")
        eval_run_id = f"ahl-eval-{_safe(s.run_id)}-{stamp}"
        model_name = f"ahl-{_safe(s.run_id)}"

        patch_text = s.patch_path.read_text(encoding="utf-8")
        predictions = work / "predictions.jsonl"
        write_text_atomic(
            predictions,
            json.dumps(
                {
                    "instance_id": s.instance_id,
                    "model_name_or_path": model_name,
                    "model_patch": patch_text,
                }
            )
            + "\n",
        )
        argv = [
            str(ctx.python), str(cli_path(ctx.python)), "eval", DATASET,
            "-p", str(predictions), "-i", s.instance_id, "--run-id", eval_run_id,
            "--task-repo", str(ctx.repository.path), "-j", str(WORKERS),
            "-t", str(INSTANCE_TIMEOUT_SECONDS), "--report-dir", str(work),
        ]  # fmt: skip
        command = EvaluatorCommand(
            argv=argv, cwd=str(work), eval_run_id=eval_run_id, model_name_or_path=model_name
        )

        # The evaluator force-rebuilds and re-tags the shared task image. Pin the original
        # first; if that is impossible, do not start (the tag would be unprotected).
        guard = ImageGuard(ctx.docker, s.docker_image, f"{_safe(s.run_id)}-{stamp}")
        try:
            guard.snapshot()
        except DockerError as exc:
            raise EvaluationError(
                f"cannot protect the task image tag {s.docker_image}: {exc}; nothing was evaluated"
            ) from exc

        warnings: list[str] = []
        stdout_log = ctx.output_dir / "evaluator.stdout.log"
        stderr_log = ctx.output_dir / "evaluator.stderr.log"
        error: str | None = None
        interrupted: BaseException | None = None
        result = None
        try:
            result = ctx.runner.run(
                argv,
                timeout=ctx.process_timeout_seconds,
                stdout_path=stdout_log,
                stderr_path=stderr_log,
                cwd=work,
                env=scrubbed_env(ctx.environ),  # no agent credentials reach the evaluator
            )
            command.exit_code, command.timed_out = result.returncode, result.timed_out
        except OSError as exc:
            error = f"could not start the evaluator: {exc}"
        except KeyboardInterrupt as exc:
            interrupted = exc
            error = "evaluation interrupted before the evaluator finished"
        finally:
            image = guard.restore()  # ALWAYS: success, failure and interruption
        finished = ctx.now()

        # Is the evaluator image really at the task baseline? Inspect THAT image by id (never the
        # restored shared tag). Skipped, and recorded as unverified, if interrupted.
        expected_base = ctx.benchmark.get_task(s.instance_id).base_commit
        if interrupted is not None:
            baseline = EvaluatorImageBaseline(
                verification_status="unverified",
                image_id=image.evaluator_image_id,
                expected_base_commit=expected_base,
                reason=REASON_UNVERIFIED,
                error="not checked: the evaluation was interrupted",
            )
        else:
            baseline = verify_baseline(
                ctx.docker,
                image.evaluator_image_id or image.image_id_before,
                expected_base,
                workdir=ctx.benchmark.container_workdir,
                label=f"{_safe(s.run_id)}-{stamp}"[:80],
            )
        guard.release_evaluator_image(image, keep=ctx.keep_evaluator_image)

        if result is None:
            state, official = EvaluationState.INFRA_FAILURE, {}
        else:
            instance_dir = (
                work / "logs" / "run_evaluation" / eval_run_id / model_name / s.instance_id
            )
            state, official, more, cls_error = classify_report(
                _read_json(work / f"{model_name}.{eval_run_id}.json"),
                s.instance_id,
                exit_code=command.exit_code,
                timed_out=command.timed_out,
                instance_report=_read_json(instance_dir / "report.json"),
                instance_log=_read_text(instance_dir / "run_instance.log"),
            )
            warnings += more
            error = cls_error or error

        reason: str | None = None
        if baseline.verification_status != "verified" and state in _MODEL_ATTRIBUTABLE:
            # Never trust (or count against the model) a verdict produced from an image whose
            # baseline is wrong or unknown: an unappliable patch may just be a wrong checkout.
            official["state_before_baseline_check"] = state.value
            state, reason = EvaluationState.INFRA_FAILURE, baseline.reason
            error = (
                f"{reason}: {baseline.error}; the official result "
                f"({official['state_before_baseline_check']}) is not attributable to the model"
            )

        if image.rebuilt:
            warnings.append(
                f"the official evaluator REBUILT {s.docker_image}: the tests ran in image "
                f"{image.evaluator_image_id} (task tag before: {image.image_id_before})"
            )
        if s.agent_image_id and image.image_id_before and s.agent_image_id != image.image_id_before:
            warnings.append(
                f"the task tag already differed from the agent's image at evaluation start "
                f"(agent ran in {s.agent_image_id}; tag was {image.image_id_before})"
            )
        warnings += image.warnings

        artifacts = {
            str(p.relative_to(ctx.output_dir)): sha256_file(p)
            for p in sorted(ctx.output_dir.rglob("*"))
            if p.is_file() and p.name != "evaluation.json" and ".superseded" not in str(p)
        }
        doc = EvaluationDocument(
            run_id=s.run_id,
            benchmark=s.benchmark,
            evaluator=self.name,
            evaluator_version=pre.environment.evaluator_version,
            task={"instance_id": s.instance_id},
            started_at=started.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            finished_at=finished.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            duration_seconds=round((finished - started).total_seconds(), 3),
            state=state,
            resolved=resolved_for(state),
            classification=classify(state),
            patch_sha256=sha256_file(s.patch_path),
            task_repository_commit=ctx.repository.commit,
            run_task_repository_commit=s.task_repository_commit,
            image=ImageEvidence(
                reference=s.docker_image,
                agent_image_id=s.agent_image_id,
                image_id_before=image.image_id_before,
                evaluator_image_id=image.evaluator_image_id,
                final_image_id=image.final_image_id,
                rebuilt=image.rebuilt,
                restoration=image.restoration,
                snapshot_ref=image.snapshot_ref,
                snapshot_removed=image.snapshot_removed,
                evaluator_image_ref=image.evaluator_image_ref,
                retention=image.retention,
                keep_requested=ctx.keep_evaluator_image,
            ),
            environment=pre.environment,
            command=command,
            evaluator_inputs_sha256=pre.evaluator_inputs_sha256,
            official_result=official,
            evaluator_build_inputs_sha256=pre.evaluator_build_inputs_sha256,
            evaluator_image_baseline=baseline,
            reason=reason,
            artifacts=artifacts,
            error=error,
            warnings=warnings,
            interrupted=interrupted is not None,
        )
        if interrupted is not None:
            doc = doc.model_copy(update={"state": EvaluationState.INFRA_FAILURE, "resolved": None})
        return doc


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


__all__ = ["SWEBenchMultilingualEvaluator", "classify_report", "EvaluatorProvenance"]
