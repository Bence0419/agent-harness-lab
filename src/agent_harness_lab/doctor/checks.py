"""Capability-oriented environment checks.

Checks talk to the host only through a ``SystemProbe`` so they can be tested
without real executables. Only foundational problems produce ERROR; missing
optional harnesses, providers or keys produce WARNING.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from os import environ
from pathlib import Path
from typing import Protocol

from agent_harness_lab.benchmarks.git import GitHead
from agent_harness_lab.benchmarks.swebench_multilingual import TASKS_SUBDIR
from agent_harness_lab.config import TASKS_REPO_ENV
from agent_harness_lab.doctor.models import CheckResult, DoctorReport, Status
from agent_harness_lab.harnesses.discovery import (
    discover_executable,
    parse_version,
    pinned_version,
)
from agent_harness_lab.paths import find_project_root

MIN_PYTHON = (3, 12)
COMMAND_TIMEOUT_SECONDS = 10.0

class CommandOutput(Protocol):
    returncode: int
    output: str


class _Output:
    def __init__(self, returncode: int, output: str) -> None:
        self.returncode = returncode
        self.output = output


class SystemProbe(Protocol):
    def system(self) -> str: ...
    def release(self) -> str: ...
    def python_version(self) -> tuple[int, int, int]: ...
    def which(self, name: str) -> str | None: ...
    def run(self, argv: list[str]) -> CommandOutput | None: ...
    def getenv(self, name: str) -> str | None: ...
    def read_text(self, path: str) -> str | None: ...
    def is_executable(self, path: str) -> bool: ...
    def is_dir(self, path: str) -> bool: ...
    def git_head(self, path: str) -> str | None: ...
    def project_root(self) -> str | None: ...


class HostProbe:
    """Real probe backed by the current machine."""

    def system(self) -> str:
        return platform.system()

    def release(self) -> str:
        return platform.release()

    def python_version(self) -> tuple[int, int, int]:
        v = sys.version_info
        return (v.major, v.minor, v.micro)

    def which(self, name: str) -> str | None:
        return shutil.which(name)

    def run(self, argv: list[str]) -> CommandOutput | None:
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=COMMAND_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return _Output(proc.returncode, (proc.stdout or proc.stderr).strip())

    def getenv(self, name: str) -> str | None:
        return environ.get(name)

    def read_text(self, path: str) -> str | None:
        try:
            return Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    def is_executable(self, path: str) -> bool:
        return os.path.isfile(path) and os.access(path, os.X_OK)

    def is_dir(self, path: str) -> bool:
        return os.path.isdir(path)

    def git_head(self, path: str) -> str | None:
        return GitHead().head_commit(Path(path))

    def project_root(self) -> str | None:
        root = find_project_root()
        return str(root) if root else None


def _ok(name: str, detail: str = "") -> CheckResult:
    return CheckResult(name=name, status=Status.OK, detail=detail)


def _warn(name: str, detail: str) -> CheckResult:
    return CheckResult(name=name, status=Status.WARNING, detail=detail)


def _error(name: str, detail: str) -> CheckResult:
    return CheckResult(name=name, status=Status.ERROR, detail=detail)


def check_os(probe: SystemProbe) -> CheckResult:
    system = probe.system()
    if system == "Linux":
        return _ok("Operating system", f"Linux {probe.release()}")
    return _warn("Operating system", f"{system or 'unknown'} is not a supported platform")


def check_wsl(probe: SystemProbe) -> CheckResult:
    version_text = (probe.read_text("/proc/version") or "").lower()
    in_wsl = bool(probe.getenv("WSL_DISTRO_NAME")) or "microsoft" in version_text
    if not in_wsl:
        return _ok("WSL", "not running under WSL")
    kind = "WSL2" if "wsl2" in f"{probe.release()} {version_text}".lower() else "WSL"
    return _ok("WSL", f"running under {kind}")


def check_python(probe: SystemProbe) -> CheckResult:
    major, minor, micro = probe.python_version()
    label = f"{major}.{minor}.{micro}"
    required = ".".join(map(str, MIN_PYTHON))
    if (major, minor) >= MIN_PYTHON:
        return _ok(f"Python >={required}", label)
    return _error(f"Python >={required}", f"found {label}, need >={required}")


def _executable_check(
    probe: SystemProbe,
    label: str,
    executable: str,
    *,
    required: bool,
    missing_detail: str,
    version_args: tuple[str, ...] = ("--version",),
) -> CheckResult:
    path = probe.which(executable)
    if path is None:
        make = _error if required else _warn
        return make(label, missing_detail)
    result = probe.run([path, *version_args])
    if result is None or result.returncode != 0:
        make = _error if required else _warn
        return make(label, f"found at {path} but `{' '.join(version_args)}` failed")
    return _ok(label, parse_version(result.output) or "found (version unknown)")


def check_git(probe: SystemProbe) -> CheckResult:
    return _executable_check(
        probe, "Git", "git", required=True, missing_detail="not found on PATH"
    )


def check_docker_cli(probe: SystemProbe) -> CheckResult:
    return _executable_check(
        probe, "Docker CLI", "docker", required=True, missing_detail="not found on PATH"
    )


def check_docker_daemon(probe: SystemProbe) -> CheckResult:
    docker = probe.which("docker")
    if docker is None:
        return _warn("Docker daemon", "not checked: Docker CLI missing")
    result = probe.run([docker, "info", "--format", "{{.ServerVersion}}"])
    if result is None or result.returncode != 0:
        return _error("Docker daemon", "not reachable (is Docker running and accessible?)")
    return _ok("Docker daemon", parse_version(result.output) or "reachable")


def check_node(probe: SystemProbe) -> CheckResult:
    return _executable_check(
        probe,
        "Node",
        "node",
        required=False,
        missing_detail="optional: not found on PATH",
    )


def check_timeout(probe: SystemProbe) -> CheckResult:
    path = probe.which("timeout")
    if path is None:
        return _warn("timeout command", "not found on PATH (GNU coreutils)")
    return _ok("timeout command", "found")


@dataclass(frozen=True)
class HarnessSpec:
    label: str
    executable: str
    npm_package: str


PI = HarnessSpec("Pi", "pi", "@earendil-works/pi-coding-agent")
OPENCODE = HarnessSpec("OpenCode", "opencode", "@opencode/cli")


def _check_harness(probe: SystemProbe, spec: HarnessSpec) -> CheckResult:
    found = discover_executable(probe, spec.executable)
    if found is None:
        return _warn(
            spec.label,
            "not installed (not found on PATH or in project node_modules/.bin)",
        )
    path, source = found.path, found.source
    result = probe.run([path, "--version"])
    if result is None or result.returncode != 0:
        return _warn(spec.label, f"found at {path} ({source}) but `--version` failed")

    version = parse_version(result.output)
    pinned = pinned_version(probe, spec.npm_package)
    if pinned is not None and version != pinned:
        return _warn(
            spec.label,
            f"{version or 'unknown version'} ({source}) differs from repository pin {pinned}",
        )
    return _ok(spec.label, f"{version or 'found (version unknown)'} ({source})")


def check_pi(probe: SystemProbe) -> CheckResult:
    return _check_harness(probe, PI)


def check_opencode(probe: SystemProbe) -> CheckResult:
    return _check_harness(probe, OPENCODE)


def check_openai_key(probe: SystemProbe) -> CheckResult:
    # Presence only: the value is never read into a result.
    if probe.getenv("OPENAI_API_KEY"):
        return _ok("OPENAI_API_KEY", "set")
    return _warn("OPENAI_API_KEY", "not set")


def check_swebench_tasks_repo(probe: SystemProbe) -> CheckResult:
    """Optional capability: never ERROR, whether unset, missing or misconfigured."""
    name = "SWE-bench tasks repo"
    raw = (probe.getenv(TASKS_REPO_ENV) or "").strip()
    if not raw:
        return _warn(name, f"not configured (set {TASKS_REPO_ENV})")
    if not probe.is_dir(raw):
        return _warn(name, f"{TASKS_REPO_ENV} points to a missing directory: {raw}")
    if not probe.is_dir(f"{raw}/{TASKS_SUBDIR}"):
        return _warn(name, f"{raw} does not look like a task repository (no {TASKS_SUBDIR}/)")
    commit = probe.git_head(raw)
    if commit is None:
        return _warn(name, f"{raw}: git commit unreadable (not a git checkout?)")
    return _ok(name, f"commit {commit}")


ALL_CHECKS: tuple[Callable[[SystemProbe], CheckResult], ...] = (
    check_os,
    check_wsl,
    check_python,
    check_git,
    check_docker_cli,
    check_docker_daemon,
    check_node,
    check_timeout,
    check_pi,
    check_opencode,
    check_openai_key,
    check_swebench_tasks_repo,
)


def run_checks(probe: SystemProbe | None = None) -> DoctorReport:
    probe = probe or HostProbe()
    return DoctorReport(results=[check(probe) for check in ALL_CHECKS])
