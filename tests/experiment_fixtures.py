"""Shared fixtures for experiment tests: fake host/Docker/evaluator, no LLM, no real Docker."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from agent_harness_lab.cli import benchmark as bench_cli
from agent_harness_lab.cli import common
from agent_harness_lab.evaluation.environment import PROBE
from agent_harness_lab.execution.gate import GATE_ENV
from agent_harness_lab.execution.models import ExecOutput
from agent_harness_lab.execution.network import SQUID_IMAGE
from tests.benchmark_fixtures import COMMIT
from tests.eval_fixtures import FakeEvaluatorRunner, make_python
from tests.fake_docker import FakeDocker
from tests.plan_fixtures import PROJECT_ROOT

SECRET = "sk-experiment-secret-value-123456"
NODE_BIN = "/opt/hostnode/v22.1.0/bin/node"
PI_REAL = f"{PROJECT_ROOT}/node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js"
OC_REAL = f"{PROJECT_ROOT}/node_modules/@opencode/cli/bin/opencode.exe"
TASKS = ("apache__lucene-12212", "axios__axios-5892", "babel__babel-15445")  # sorted


def image_of(task: str) -> str:
    return f"swebench/sweb.eval.x86_64.{task.replace('__', '_1776_')}:latest"


class FakeHost:
    def __init__(self, missing=()):
        self.missing = set(missing)
        self.links = {"/fake/pi": PI_REAL, "/fake/opencode": OC_REAL}

    def which(self, name):
        return NODE_BIN if name == "node" else None

    def realpath(self, path):
        return self.links.get(path, path)

    def is_file(self, path):
        return path not in self.missing

    def is_dir(self, path):
        return path not in self.missing


class ScriptedDocker(FakeDocker):
    """FakeDocker whose harness outcome depends on the (task, harness) of the container."""

    def __init__(self, *a, script=None, interrupt_on=None, **kw):
        super().__init__(*a, **kw)
        self.script = script or {}  # (task, harness) -> "fail" | "timeout"
        self.interrupt_on = interrupt_on  # (task, harness) -> raise KeyboardInterrupt
        self.agent_runs: list[tuple[str, str]] = []

    def exec(self, container, argv, **kw):
        is_agent = not (
            argv[0] in ("test", "git", "mkdir") or "-C" in argv or argv[:2] == ["git", "-c"]
        )
        if not is_agent:
            return super().exec(container, argv, **kw)
        task, harness = self._key(container)
        self.agent_runs.append((task, harness))
        if self.interrupt_on == (task, harness):
            raise KeyboardInterrupt
        mode = self.script.get((task, harness))
        self.harness_rc = 3 if mode == "fail" else 0
        self.harness_timed_out = mode == "timeout"
        return super().exec(container, argv, **kw)

    @staticmethod
    def _key(container: str) -> tuple[str, str]:
        name = container.removeprefix("ahl-agent-")
        for task in TASKS:
            if name.startswith(task + "-"):
                return task, name[len(task) + 1 :].split("-")[0]
        raise AssertionError(container)

    def created_containers(self) -> int:
        return self.ops().count("create_container")


class PerTaskEvaluator(FakeEvaluatorRunner):
    """Evaluator whose official bucket depends on the instance being scored."""

    def __init__(self, outcomes=None, **kw):
        super().__init__(**kw)
        self.outcomes = outcomes or {}
        self.evaluated: list[str] = []
        self.probes = 0
        self.break_after_probes: int | None = None  # environment probe fails after N successes

    def run(self, argv, **kw):
        if argv[1:3] == ["-c", PROBE]:
            self.probes += 1
            if self.break_after_probes is not None and self.probes > self.break_after_probes:
                self.probe_returncode = 1
        if "-i" in argv:
            instance = argv[argv.index("-i") + 1]
            self.evaluated.append(instance)
            self.bucket = self.outcomes.get(instance, "resolved_ids")
        return super().run(argv, **kw)


def experiment_yaml(
    *,
    tasks=("babel__babel-15445",),
    harnesses=("pi",),
    models=(("openai", "gpt-6-astra", "medium"),),
    repetitions=1,
    evaluation=True,
    pin=COMMIT,
    name="astra-test",
    timeout=3600,
    network="restricted-provider",
    extra="",
) -> str:
    lines = [
        "schema_version: ahl-experiment-v1",
        "experiment:",
        f"  name: {name}",
        "  description: test experiment",
        "  purpose: test",
        "benchmark:",
        "  name: swebench-multilingual",
    ]
    if pin:
        lines.append(f"  task_repository_commit: {pin}")
    lines.append("tasks:")
    lines += [f"  - {t}" for t in tasks]
    lines.append("harnesses:")
    lines += [f"  - {h}" for h in harnesses]
    lines.append("models:")
    for provider, model, reasoning in models:
        lines += [f"  - provider: {provider}", f"    model: {model}"]
        if reasoning:
            lines.append(f"    reasoning: {reasoning}")
    lines += [
        f"repetitions: {repetitions}",
        "execution:",
        f"  timeout_seconds: {timeout}",
        f"  network: {network}",
        "evaluation:",
        f"  enabled: {'true' if evaluation else 'false'}",
    ]
    return "\n".join(lines) + "\n" + textwrap.dedent(extra)


@pytest.fixture
def exp_env(monkeypatch, plan_probe, plan_git, task_repo, tmp_path):
    """Everything an ``ahl benchmark`` invocation touches, replaced by fakes."""
    plan_probe.env["OPENAI_API_KEY"] = SECRET
    python = make_python(tmp_path)
    docker = ScriptedDocker(images={*(image_of(t) for t in TASKS), SQUID_IMAGE})
    state = {
        "env": {"OPENAI_API_KEY": SECRET, GATE_ENV: "1"},
        "docker": docker,
        "host": FakeHost(),
        "evaluator": PerTaskEvaluator(docker=docker),
        "raw": tmp_path / "raw",
        "experiments": tmp_path / "experiments",
        "python": python,
        "tasks_repo": task_repo,
        "tmp": tmp_path,
        "git": plan_git,
        "probe": plan_probe,
    }
    monkeypatch.setattr(bench_cli, "_probe", lambda: plan_probe)
    monkeypatch.setattr(bench_cli, "_git", lambda: plan_git)
    monkeypatch.setattr(common, "environ", lambda: state["env"])
    monkeypatch.setattr(common, "docker_client", lambda: state["docker"])
    monkeypatch.setattr(common, "host_environment", lambda: state["host"])
    monkeypatch.setattr(common, "command_runner", lambda: state["evaluator"])
    monkeypatch.setattr(common, "sleep", lambda s: None)
    monkeypatch.setenv("AHL_SWEBENCH_TASKS_REPO", str(task_repo))
    monkeypatch.setenv("AHL_SWEBENCH_PYTHON", str(python))
    return state


def write_experiment(state, text: str, name="experiment.yaml") -> Path:
    path = state["tmp"] / name
    path.write_text(text, encoding="utf-8")
    return path


def experiment_dirs(state) -> list[Path]:
    root: Path = state["experiments"]
    return sorted(p for p in root.iterdir()) if root.is_dir() else []


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def no_exec(argv):  # evaluator calls that are real evaluator runs (not the environment probe)
    return "-i" in argv


def exec_output(rc=0):
    return ExecOutput(returncode=rc)
