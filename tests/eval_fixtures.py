"""Helpers for evaluation tests: a fake evaluator process that writes a report in the official
SWE-bench report schema, plus runs created through the real executor (with FakeDocker)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from agent_harness_lab.benchmarks.swebench_multilingual import SWEBenchMultilingual
from agent_harness_lab.evaluation.base import EvaluationContext
from agent_harness_lab.evaluation.environment import PROBE
from agent_harness_lab.evaluation.service import load_subject
from agent_harness_lab.execution.docker import RunResult
from agent_harness_lab.execution.executor import RunRequest, execute_run
from tests.benchmark_fixtures import COMMIT, FakeGit
from tests.fake_docker import FakeDocker
from tests.mock_harness import mock_plan, mock_provisioner, write_mock_harness

TASK = "babel__babel-15445"
TASK_IMAGE = "swebench/sweb.eval.x86_64.babel_1776_babel-15445:latest"
KEY = "sk-eval-secret-must-not-reach-the-evaluator"


def report_for(
    instance: str,
    bucket: str | None,
    *,
    completed: bool = True,
    tags: tuple[str, ...] = (),
    reason: str | None = None,
    extra=None,
) -> dict:
    """An official-style report (schema 2): ``instance`` in one PRIMARY list; ``tags`` are the
    additive infra/ambiguous lists the official reporter adds over unresolved/error."""
    lists = {
        k: []
        for k in (
            "completed_ids",
            "incomplete_ids",
            "empty_patch_ids",
            "submitted_ids",
            "resolved_ids",
            "unresolved_ids",
            "infra_failure_ids",
            "ambiguous_failure_ids",
            "error_ids",
        )
    }
    lists["submitted_ids"] = [instance]
    if completed:
        lists["completed_ids"] = [instance]
    if bucket:
        lists[bucket] = [*lists[bucket], instance]
    for tag in tags:
        lists[tag] = [*lists[tag], instance]
    return {
        "total_instances": 1,
        "schema_version": 2,
        "failure_reasons": {instance: reason} if reason else {},
        **lists,
        **(extra or {}),
    }


class FakeEvaluatorRunner:
    """Stands in for the SWE-bench CLI process and the environment probe."""

    def __init__(
        self,
        *,
        bucket="resolved_ids",
        report="auto",
        returncode=0,
        timed_out=False,
        probe_returncode=0,
        version="5.0.2",
        start_error=None,
        extra_buckets=(),
        tags=(),
        reason=None,
        log_text=None,
        instance_report=None,
        docker=None,
        rebuild_to=None,
        interrupt=False,
    ):
        self.bucket, self.report, self.returncode, self.timed_out = (
            bucket,
            report,
            returncode,
            timed_out,
        )
        self.probe_returncode, self.version, self.start_error = (
            probe_returncode,
            version,
            start_error,
        )
        self.extra_buckets = extra_buckets
        self.tags, self.reason, self.log_text, self.instance_report = (
            tags,
            reason,
            log_text,
            instance_report,
        )
        self.docker, self.rebuild_to, self.interrupt = docker, rebuild_to, interrupt
        self.calls: list[dict] = []

    def run(
        self,
        argv,
        *,
        timeout=None,
        stdout_path=None,
        stderr_path=None,
        merge_stderr=False,
        cwd=None,
        env=None,
    ):
        self.calls.append({"argv": list(argv), "timeout": timeout, "cwd": cwd, "env": env})
        if argv[1:3] == ["-c", PROBE]:
            if self.probe_returncode:
                return RunResult(
                    self.probe_returncode, stderr="ModuleNotFoundError: No module named 'swebench'"
                )
            info = {"executable": argv[0], "python": "3.12.3", "swebench": self.version}
            return RunResult(0, stdout=json.dumps(info) + "\n")
        if self.start_error:
            raise OSError(self.start_error)
        if self.docker is not None and self.rebuild_to:  # the evaluator force-rebuilds the image
            self.docker.external_rebuild(TASK_IMAGE, self.rebuild_to)
        if self.interrupt:
            raise KeyboardInterrupt
        pred = json.loads(Path(argv[argv.index("-p") + 1]).read_text())
        instance, run_id = argv[argv.index("-i") + 1], argv[argv.index("--run-id") + 1]
        report_dir = Path(argv[argv.index("--report-dir") + 1])
        if stdout_path:
            stdout_path.write_text("evaluator stdout\n")
        if stderr_path:
            stderr_path.write_text("")
        if self.report == "auto":
            report = report_for(instance, self.bucket, tags=self.tags, reason=self.reason)
            for extra in self.extra_buckets:
                report[extra] = [instance]
        else:
            report = self.report
        if report is not None:
            (report_dir / f"{pred['model_name_or_path']}.{run_id}.json").write_text(
                json.dumps(report)
            )
        inst_dir = (
            report_dir / "logs" / "run_evaluation" / run_id / pred["model_name_or_path"] / instance
        )
        inst_dir.mkdir(parents=True, exist_ok=True)
        (inst_dir / "run_instance.log").write_text(self.log_text or "INFO - evaluation ran\n")
        if self.instance_report is not None:
            (inst_dir / "report.json").write_text(json.dumps({instance: self.instance_report}))
        return RunResult(self.returncode, timed_out=self.timed_out)


class Clock:
    def __init__(self, start=datetime(2026, 11, 24, 12, 0, 0, tzinfo=UTC), step=1):
        self.t, self.step = start, step

    def __call__(self):
        self.t += timedelta(seconds=self.step)
        return self.t


def make_ahl_run(
    tmp_path: Path, raw: Path, *, harness_name="mock", patch_text=None, **plan_kw
) -> Path:
    """A finished ahl-run-v1 run directory produced by the real executor against FakeDocker."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    plan = mock_plan(
        benchmark="swebench-multilingual",
        task_id=TASK,
        image=TASK_IMAGE,
        prompt="p",
        harness_name=harness_name,
        task_repository_commit=COMMIT,
        **plan_kw,
    )
    docker = (
        FakeDocker(images={TASK_IMAGE})
        if patch_text is None
        else FakeDocker(images={TASK_IMAGE}, patch_text=patch_text)
    )
    ticks = Clock(datetime(2026, 11, 24, 10, 0, 0, tzinfo=UTC))
    doc = execute_run(
        RunRequest(
            plan=plan,
            prompt="p",
            provisioner=mock_provisioner(write_mock_harness(tmp_path)),
            raw_dir=raw,
        ),
        docker=docker,
        now=ticks,
        sleep=lambda s: None,
    )
    return raw / "swebench" / TASK / doc.run_id


def make_ctx(
    run_dir: Path,
    task_repo: Path,
    python: Path | None,
    runner,
    *,
    docker=None,
    output_dir=None,
    force=False,
    environ=None,
    git=None,
    step=1,
    keep_evaluator_image=False,
):
    benchmark = SWEBenchMultilingual(task_repo, git or FakeGit())
    return EvaluationContext(
        subject=load_subject(run_dir),
        benchmark=benchmark,
        repository=benchmark.repository_info(),
        python=python,
        docker=docker or FakeDocker(images={TASK_IMAGE}),
        runner=runner,
        environ=environ if environ is not None else {},
        output_dir=output_dir or run_dir,
        force=force,
        keep_evaluator_image=keep_evaluator_image,
        now=Clock(step=step),
    )


def make_python(tmp_path: Path) -> Path:
    """A stand-in interpreter plus the console script the evaluator installs next to it."""
    bin_dir = tmp_path / "evalenv" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "python").write_text("")
    (bin_dir / "swebench").write_text("")
    return bin_dir / "python"
