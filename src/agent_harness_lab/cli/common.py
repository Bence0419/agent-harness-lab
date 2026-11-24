"""Injectable process-level seams for the CLI (tests replace these; never import-bind them)."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping

from agent_harness_lab.execution.docker import (
    CommandRunner,
    DockerCli,
    DockerClient,
    SubprocessRunner,
)
from agent_harness_lab.harnesses.provisioning import HostEnvironment, RealHost


def environ() -> Mapping[str, str]:
    return os.environ


def docker_client() -> DockerClient:
    return DockerCli()


def host_environment() -> HostEnvironment:
    return RealHost()


def sleep(seconds: float) -> None:
    time.sleep(seconds)


def command_runner() -> CommandRunner:
    return SubprocessRunner()
