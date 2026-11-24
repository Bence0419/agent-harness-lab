"""Docker access through argv lists only (never shell strings), behind injectable seams.

``CommandRunner`` executes one argv; ``DockerCli`` turns Docker operations into argv. Tests
substitute a recording runner (argv construction) or an in-memory ``DockerClient`` (executor).
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from agent_harness_lab.execution.models import (
    ContainerSpec,
    DockerError,
    ExecOutput,
    ImageMissing,
    ImageRecord,
    Provision,
)

KILL_AFTER_SECONDS = 30.0  # frozen runners: `timeout --signal=TERM --kill-after=30s`


@dataclass(frozen=True)
class RunResult:
    returncode: int | None
    timed_out: bool = False
    stdout: str = ""
    stderr: str = ""


class CommandRunner(Protocol):
    def run(
        self,
        argv: list[str],
        *,
        timeout: float | None = None,
        stdout_path: Path | None = None,
        stderr_path: Path | None = None,
        merge_stderr: bool = False,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> RunResult: ...


class SubprocessRunner:
    """Real runner. Inherits the process environment, so ``-e NAME`` forwards secrets from it."""

    def run(
        self,
        argv: list[str],
        *,
        timeout: float | None = None,
        stdout_path: Path | None = None,
        stderr_path: Path | None = None,
        merge_stderr: bool = False,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> RunResult:
        out_f = stdout_path.open("wb") if stdout_path else None
        err_f = stderr_path.open("wb") if stderr_path else None
        try:
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=out_f or subprocess.PIPE,
                stderr=subprocess.STDOUT if merge_stderr else (err_f or subprocess.PIPE),
                cwd=cwd,
                env=None if env is None else dict(env),
            )
            timed_out = False
            try:
                out, err = proc.communicate(timeout=timeout)
            except BaseException as exc:
                if not isinstance(exc, subprocess.TimeoutExpired):
                    # interrupted (e.g. Ctrl-C): never leave the child running unattended
                    proc.terminate()
                    try:
                        proc.wait(timeout=KILL_AFTER_SECONDS)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    raise
                timed_out = True
                proc.terminate()  # SIGTERM first, SIGKILL after the grace period
                try:
                    out, err = proc.communicate(timeout=KILL_AFTER_SECONDS)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    out, err = proc.communicate()
        finally:
            for f in (out_f, err_f):
                if f:
                    f.close()
        return RunResult(
            proc.returncode,
            timed_out,
            out.decode("utf-8", "replace") if out else "",
            err.decode("utf-8", "replace") if err else "",
        )


class DockerClient(Protocol):
    def image_inspect(self, reference: str) -> ImageRecord: ...
    def network_create(self, name: str, *, internal: bool, labels: dict[str, str]) -> None: ...
    def network_connect(self, network: str, container: str) -> None: ...
    def network_remove(self, name: str) -> None: ...
    def run_proxy(
        self,
        *,
        name: str,
        network: str,
        alias: str,
        image: str,
        mounts: list[Provision],
        labels: dict[str, str],
    ) -> None: ...
    def create_container(self, spec: ContainerSpec) -> None: ...
    def start(self, name: str) -> None: ...
    def kill(self, name: str) -> None: ...
    def inspect_container(self, name: str) -> dict: ...
    def copy_in(self, source: Path, container: str, destination: str) -> None: ...
    def exec(
        self,
        container: str,
        argv: list[str],
        *,
        workdir: str | None = None,
        timeout: float | None = None,
        stdout_path: Path | None = None,
        stderr_path: Path | None = None,
    ) -> ExecOutput: ...
    def logs(self, name: str) -> str: ...
    def remove(self, *names: str) -> None: ...
    def tag_image(self, source: str, target: str) -> None: ...
    def untag_image(self, reference: str) -> None: ...
    def list_image_ids(self) -> set[str]: ...
    def remove_image(self, image_id: str) -> None: ...


def _mount_arg(m: Provision) -> str:
    return f"{m.source}:{m.destination}" + (":ro" if m.read_only else "")


def _label_args(labels: dict[str, str]) -> list[str]:
    return [a for k, v in sorted(labels.items()) for a in ("--label", f"{k}={v}")]


class DockerCli:
    def __init__(self, runner: CommandRunner | None = None, docker: str = "docker") -> None:
        self._runner = runner or SubprocessRunner()
        self._docker = docker

    def _checked(self, argv: list[str], what: str) -> RunResult:
        result = self._runner.run(argv)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip().splitlines()
            raise DockerError(
                f"{what} failed (exit {result.returncode}): {detail[-1] if detail else ''}"
            )
        return result

    # ---- images ------------------------------------------------------------------
    def image_inspect(self, reference: str) -> ImageRecord:
        result = self._runner.run([self._docker, "image", "inspect", reference])
        if result.returncode != 0:
            raise ImageMissing(
                f"Docker image not found locally: {reference} "
                "(automatic pull/build is not implemented)"
            )
        try:
            info = json.loads(result.stdout)[0]
        except (ValueError, IndexError, KeyError) as exc:
            raise DockerError(f"unreadable image inspect output for {reference}") from exc
        return ImageRecord(
            reference=reference,
            image_id=info.get("Id"),
            repo_digests=list(info.get("RepoDigests") or []),
        )

    def list_image_refs(self) -> set[str]:
        """Local ``repository:tag`` references (one read-only call; used for candidate checks)."""
        result = self._checked(
            [self._docker, "images", "--format", "{{.Repository}}:{{.Tag}}"], "list images"
        )
        return {line.strip() for line in result.stdout.splitlines() if line.strip()}

    # ---- networks ------------------------------------------------------------------
    def network_create(self, name: str, *, internal: bool, labels: dict[str, str]) -> None:
        argv = [self._docker, "network", "create"]
        if internal:
            argv.append("--internal")
        self._checked([*argv, *_label_args(labels), name], f"network create {name}")

    def network_connect(self, network: str, container: str) -> None:
        self._checked([self._docker, "network", "connect", network, container], "network connect")

    def network_remove(self, name: str) -> None:
        self._checked([self._docker, "network", "rm", name], f"network rm {name}")

    def run_proxy(self, *, name, network, alias, image, mounts, labels) -> None:
        argv = [
            self._docker, "run", "-d", "--name", name, "--network", network,
            "--network-alias", alias, *_label_args(labels),
        ]  # fmt: skip
        for m in mounts:
            argv += ["-v", _mount_arg(m)]
        self._checked([*argv, image], f"start proxy {name}")

    # ---- containers --------------------------------------------------------------------
    def create_container(self, spec: ContainerSpec) -> None:
        argv = [self._docker, "create", "--name", spec.name, "--entrypoint", "sleep"]
        argv += _label_args(spec.labels)
        if spec.network:
            argv += ["--network", spec.network]
        for k, v in spec.env.items():
            argv += ["-e", f"{k}={v}"]
        for name in spec.pass_env:  # value comes from the docker CLI's own environment
            argv += ["-e", name]
        for m in spec.mounts:
            argv += ["-v", _mount_arg(m)]
        self._checked([*argv, spec.image, "infinity"], f"create container {spec.name}")

    def start(self, name: str) -> None:
        self._checked([self._docker, "start", name], f"start {name}")

    def kill(self, name: str) -> None:
        self._checked([self._docker, "kill", name], f"kill {name}")

    def inspect_container(self, name: str) -> dict:
        result = self._checked([self._docker, "container", "inspect", name], f"inspect {name}")
        try:
            return json.loads(result.stdout)[0]
        except (ValueError, IndexError) as exc:
            raise DockerError(f"unreadable container inspect output for {name}") from exc

    def copy_in(self, source: Path, container: str, destination: str) -> None:
        self._checked(
            [self._docker, "cp", str(source), f"{container}:{destination}"],
            f"copy to {destination}",
        )

    def exec(
        self,
        container: str,
        argv: list[str],
        *,
        workdir: str | None = None,
        timeout: float | None = None,
        stdout_path: Path | None = None,
        stderr_path: Path | None = None,
    ) -> ExecOutput:
        cmd = [self._docker, "exec"]
        if workdir:
            cmd += ["-w", workdir]
        result = self._runner.run(
            [*cmd, container, *argv],
            timeout=timeout,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        return ExecOutput(
            returncode=result.returncode,
            timed_out=result.timed_out,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    def logs(self, name: str) -> str:
        result = self._runner.run([self._docker, "logs", name], merge_stderr=True)
        if result.returncode != 0:
            raise DockerError(f"logs {name} failed (exit {result.returncode})")
        return result.stdout

    def remove(self, *names: str) -> None:
        self._checked([self._docker, "rm", "-f", *names], "remove containers")

    # ---- image tags (used to protect shared task-image tags during evaluation) -------------
    def tag_image(self, source: str, target: str) -> None:
        """``docker tag``: point ``target`` at the image ``source`` (an id or a reference)."""
        self._checked([self._docker, "tag", source, target], f"tag {target}")

    def list_image_ids(self) -> set[str]:
        """Ids of ALL local images (tagged or not), to know what existed before an evaluation."""
        result = self._checked(
            [self._docker, "images", "-a", "--no-trunc", "--format", "{{.ID}}"], "list image ids"
        )
        return {line.strip() for line in result.stdout.splitlines() if line.strip()}

    def remove_image(self, image_id: str) -> None:
        """``docker rmi <id>`` WITHOUT -f: Docker refuses if any tag still references the image
        or a container uses it, so it can never remove something still in use."""
        self._checked([self._docker, "rmi", image_id], f"remove image {image_id[:19]}")

    def untag_image(self, reference: str) -> None:
        """``docker rmi <reference>`` WITHOUT -f: removes one reference. If it is the image's
        last reference Docker would delete the image, so callers must only untag references
        that an AHL snapshot guarantees are not the last."""
        self._checked([self._docker, "rmi", reference], f"untag {reference}")
