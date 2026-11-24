import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from agent_harness_lab.doctor.checks import _Output
from tests.benchmark_fixtures import COMMIT, FakeGit, make_task_repo
from tests.experiment_fixtures import exp_env  # noqa: F401  (fixture shared by experiment tests)
from tests.plan_fixtures import (
    AHL_COMMIT,
    PACKAGE_JSON,
    PROJECT_ROOT,
    PROVIDERS_YAML,
    SECRET,
)


@dataclass
class FakeProbe:
    """In-memory system probe; never touches the real host."""

    os_name: str = "Linux"
    os_release: str = "6.1.0-generic"
    py: tuple[int, int, int] = (3, 12, 3)
    executables: dict[str, str] = field(default_factory=dict)  # name -> path
    commands: dict[tuple[str, ...], tuple[int, str]] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)
    executable_files: set[str] = field(default_factory=set)
    dirs: set[str] = field(default_factory=set)
    heads: dict[str, str] = field(default_factory=dict)  # repo path -> HEAD commit
    root: str | None = None
    calls: list[list[str]] = field(default_factory=list)

    def system(self) -> str:
        return self.os_name

    def release(self) -> str:
        return self.os_release

    def python_version(self) -> tuple[int, int, int]:
        return self.py

    def which(self, name: str) -> str | None:
        return self.executables.get(name)

    def run(self, argv: list[str]):
        self.calls.append(argv)
        hit = self.commands.get(tuple(argv))
        return None if hit is None else _Output(*hit)

    def getenv(self, name: str) -> str | None:
        return self.env.get(name)

    def read_text(self, path: str) -> str | None:
        return self.files.get(path)

    def is_executable(self, path: str) -> bool:
        return path in self.executable_files

    def is_dir(self, path: str) -> bool:
        return path in self.dirs

    def git_head(self, path: str) -> str | None:
        return self.heads.get(path)

    def project_root(self) -> str | None:
        return self.root


@pytest.fixture
def healthy_probe() -> FakeProbe:
    return FakeProbe(
        executables={
            "git": "/fake/git",
            "docker": "/fake/docker",
            "node": "/fake/node",
            "timeout": "/fake/timeout",
            "pi": "/fake/pi",
            "opencode": "/fake/opencode",
        },
        commands={
            ("/fake/git", "--version"): (0, "git version 2.43.0"),
            ("/fake/docker", "--version"): (0, "Docker version 27.1.1, build abc"),
            ("/fake/docker", "info", "--format", "{{.ServerVersion}}"): (0, "27.1.1"),
            ("/fake/node", "--version"): (0, "v24.19.0"),
            ("/fake/pi", "--version"): (0, "1.0.2"),
            ("/fake/opencode", "--version"): (0, "opencode v2.2.0"),
        },
        env={
            "OPENAI_API_KEY": "sk-super-secret-value",
            "AHL_SWEBENCH_TASKS_REPO": "/fake/tasks-repo",
        },
        dirs={"/fake/tasks-repo", "/fake/tasks-repo/tasks"},
        heads={"/fake/tasks-repo": "6e08cbcd763c869abd7aae4bcb9e1b3991180777"},
    )


@pytest.fixture
def plan_probe(healthy_probe):
    """Healthy fake host with a project root and repository pins (Pi 1.0.2, OpenCode 2.2.0)."""
    healthy_probe.root = PROJECT_ROOT
    healthy_probe.files[f"{PROJECT_ROOT}/package.json"] = PACKAGE_JSON
    healthy_probe.env["EXAMPLE_LAB_API_KEY"] = SECRET + "-example"
    return healthy_probe


@pytest.fixture
def plan_git():
    git = FakeGit(COMMIT, dirty=False)
    git.set_repo(PROJECT_ROOT, AHL_COMMIT, False)
    return git


@pytest.fixture
def providers_file(tmp_path: Path) -> Path:
    path = tmp_path / "providers.yaml"
    path.write_text(PROVIDERS_YAML)
    return path


@pytest.fixture
def task_repo(tmp_path: Path) -> Path:
    return make_task_repo(tmp_path)


def pytest_collection_modifyitems(config, items):
    """Docker integration tests are opt-in: `pytest -m docker` or AHL_DOCKER_TESTS=1."""
    expr = (config.getoption("-m") or "").strip()
    enabled = os.environ.get("AHL_DOCKER_TESTS") == "1" or (
        "docker" in expr and "not docker" not in expr
    )
    if enabled:
        return
    skip = pytest.mark.skip(reason="opt-in: run `pytest -m docker` or set AHL_DOCKER_TESTS=1")
    for item in items:
        if "docker" in item.keywords:
            item.add_marker(skip)
