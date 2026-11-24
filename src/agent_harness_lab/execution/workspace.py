"""Workspace preparation and patch capture, reproducing the frozen runners step by step.

Preparation (inside the task container, working directory /testbed):
  git config core.fileMode false; verify HEAD == base commit; git reset --hard <base>;
  git clean -fd; verify `git status --porcelain` is empty.
Capture: `git status --short`, then `git add -N -- .` (intent-to-add, so new files appear in
the diff; failures ignored), then `git diff --binary <base>`.
"""

from __future__ import annotations

from pathlib import Path

from agent_harness_lab.execution.docker import DockerClient
from agent_harness_lab.execution.models import StageError

GIT = ["git", "-c", "core.fileMode=false"]


def _git(docker: DockerClient, container: str, workdir: str, argv: list[str], what: str):
    out = docker.exec(container, argv, workdir=workdir)
    if out.returncode != 0:
        detail = (out.stderr or out.stdout).strip()
        raise StageError("workspace", f"{what} failed (exit {out.returncode}): {detail}")
    return out


def prepare_workspace(docker: DockerClient, container: str, base_commit: str, workdir: str) -> None:
    _git(docker, container, workdir, ["git", "config", "core.fileMode", "false"], "git config")
    head = _git(docker, container, workdir, ["git", "rev-parse", "HEAD"], "git rev-parse HEAD")
    if head.stdout.strip() != base_commit:
        raise StageError(
            "workspace",
            f"unexpected repository HEAD {head.stdout.strip()!r} (expected {base_commit})",
        )
    _git(docker, container, workdir, ["git", "reset", "--hard", base_commit], "git reset --hard")
    _git(docker, container, workdir, ["git", "clean", "-fd"], "git clean -fd")
    status = _git(docker, container, workdir, ["git", "status", "--porcelain"], "git status")
    if status.stdout.strip():
        raise StageError(
            "workspace", f"repository is not clean before agent execution:\n{status.stdout}"
        )


def capture_patch(
    docker: DockerClient, container: str, base_commit: str, workdir: str, run_dir: Path
) -> None:
    """Write git-status.txt and model.patch (the patch may legitimately be empty)."""
    status = docker.exec(
        container,
        [*GIT, "status", "--short"],
        workdir=workdir,
        stdout_path=run_dir / "git-status.txt",
    )
    if status.returncode != 0:
        raise StageError("patch", f"git status failed (exit {status.returncode})")
    docker.exec(container, ["git", "add", "-N", "--", "."], workdir=workdir)  # best effort
    diff = docker.exec(
        container,
        [*GIT, "diff", "--binary", base_commit],
        workdir=workdir,
        stdout_path=run_dir / "model.patch",
    )
    if diff.returncode != 0:
        raise StageError("patch", f"git diff failed (exit {diff.returncode})")
