"""Deterministic TEST-ONLY harness for validating the execution layer.

It needs no model, no network and no secret. It is not in the public harness registry; a
RunPlan for it is built by hand here. Behaviors (first argument ``--behavior``):
  edit  append one line to mock_target.txt, emit a JSONL trace, exit 0
  noop  change nothing, exit 0
  fail  edit, then exit 3
  sleep sleep long enough to trigger any timeout
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

from agent_harness_lab.artifacts.hashing import sha256_text
from agent_harness_lab.execution.models import Provision
from agent_harness_lab.harnesses.base import (
    PROMPT_PLACEHOLDER,
    GeneratedFile,
    HarnessInvocation,
    OutputExpectation,
)
from agent_harness_lab.providers import ModelSpec
from agent_harness_lab.runplan.models import (
    AhlRef,
    HarnessRef,
    ModelCapabilities,
    NetworkMode,
    NetworkPlan,
    PromptRef,
    ProviderRef,
    RunPlan,
    TaskRef,
)

MOCK_CONTAINER_PATH = "/opt/ahl-mock/mock-harness"
TARGET_FILE = "mock_target.txt"
EDIT_LINE = "edited by mock harness"

MOCK_SCRIPT = """#!/bin/sh
behavior=edit
if [ "$1" = "--behavior" ]; then behavior="$2"; shift 2; fi
echo '{"type":"mock_start","behavior":"'"$behavior"'"}'
case "$behavior" in
  edit) printf 'edited by mock harness\\n' >> mock_target.txt
        echo '{"type":"mock_edit","file":"mock_target.txt"}'
        echo '{"type":"mock_end"}'; exit 0 ;;
  noop) echo '{"type":"mock_end"}'; exit 0 ;;
  fail) printf 'edited by mock harness\\n' >> mock_target.txt
        echo '{"type":"mock_end"}'; echo 'mock failure' >&2; exit 3 ;;
  sleep) sleep 600 ;;
esac
"""


def write_mock_harness(directory: Path) -> Path:
    path = directory / "mock-harness"
    path.write_text(MOCK_SCRIPT, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def mock_plan(
    *,
    behavior: str = "edit",
    timeout: int | None = 3600,
    network: NetworkMode = NetworkMode.UNRESTRICTED,
    allowed_hosts: list[str] | None = None,
    image: str = "ahl-test-synthetic:1",
    base_commit: str = "a" * 40,
    prompt: str = "Fix the mock issue.",
    secret_env: list[str] | None = None,
    config_files: list[GeneratedFile] | None = None,
    harness_name: str = "mock",
    task_id: str = "mock__task-1",
    benchmark: str = "mock-benchmark",
    task_repository_commit: str = "b" * 40,
) -> RunPlan:
    return RunPlan(
        benchmark=benchmark,
        task=TaskRef(
            instance_id=task_id,
            repo="mock/repo",
            base_commit=base_commit,
            docker_image=image,
            task_repository_commit=task_repository_commit,
            task_repository_dirty=False,
        ),
        prompt=PromptRef(
            preamble="", delivered_sha256=sha256_text(prompt), delivered_chars=len(prompt)
        ),
        workdir="/testbed",
        harness=HarnessRef(
            name=harness_name,
            display_name="Mock",
            executable_path="/host/mock-harness",
            discovery_source="PATH",
            version="0.0.1",
            expected_version=None,
        ),
        invocation=HarnessInvocation(
            argv=[MOCK_CONTAINER_PATH, "--behavior", behavior, PROMPT_PLACEHOLDER],
            env={"MOCK_ENV": "1"},
            secret_env=list(secret_env or []),
            config_files=list(config_files or []),
            output=OutputExpectation(trace_format="Mock", stderr_file="mock.stderr.log"),
        ),
        provider=ProviderRef(
            id="mock",
            type="openai-compatible",
            display_name=None,
            base_url="http://mock.invalid/v1",
            base_url_env=None,
            api_key_env="MOCK_API_KEY",
        ),
        model=ModelSpec(provider="mock", model="mock-1"),
        model_capabilities=ModelCapabilities(reasoning_supported=None),
        timeout_seconds=timeout,
        network=NetworkPlan(
            mode=network,
            allowed_hosts=allowed_hosts
            if allowed_hosts is not None
            else (["mock.example.test"] if network is NetworkMode.RESTRICTED_PROVIDER else []),
        ),
        required_secret_env=list(secret_env or []),
        ahl=AhlRef(commit="c" * 40, dirty=False),
    )


def mock_provisioner(host_script: Path, extra_mounts: list[Provision] | None = None):
    """Provisioner mounting the mock executable read-only, via the same mechanism the real
    harness adapters use (structured Provision records)."""

    def provision(staged: dict[str, Path]) -> list[Provision]:
        out = [
            Provision(
                kind="bind-mount",
                source=str(host_script.resolve()),
                destination=MOCK_CONTAINER_PATH,
                read_only=True,
                purpose="mock harness executable",
            )
        ]
        for container_path, path in sorted(staged.items()):
            out.append(
                Provision(
                    kind="copy-in",
                    source=str(path),
                    destination=container_path,
                    read_only=False,
                    purpose="generated config",
                )
            )
        return [*out, *(extra_mounts or [])]

    return provision


def host_env() -> dict[str, str]:
    return dict(os.environ)
