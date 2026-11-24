"""OpenCode adapter: reproduces the frozen SWE-bench runners' OpenCode treatment.

Source of truth: scripts/run-swebench-task-opencode{,-astra,-bme-qwen}.sh:
``opencode run --standalone --auto --agent build --model <provider>/<model>[#variant]
--format json <prompt>`` with OPENCODE_CONFIG pointing at a policy file that denies
webfetch/websearch (and, for OpenAI-compatible providers, also defines the provider).

Container layout of the frozen runners (a Phase 6 provisioning contract): the resolved
opencode binary mounted read-only at /usr/local/bin/opencode and the rendered config at
/etc/opencode/opencode.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

from agent_harness_lab.execution.models import Provision
from agent_harness_lab.harnesses.base import (
    PROMPT_PLACEHOLDER,
    GeneratedFile,
    HarnessError,
    HarnessInvocation,
    OutputExpectation,
)
from agent_harness_lab.harnesses.provisioning import HostEnvironment
from agent_harness_lab.providers import ResolvedModel

CONTAINER_EXECUTABLE = "/usr/local/bin/opencode"
CONFIG_PATH = "/etc/opencode/opencode.json"
AGENT = "build"
SCHEMA_URL = "https://opencode.ai/config.json"
PERMISSIONS = [
    {"action": "webfetch", "resource": "*", "effect": "deny"},
    {"action": "websearch", "resource": "*", "effect": "deny"},
]


def render_config(model: ResolvedModel) -> str:
    """opencode.json: the benchmark policy, plus the provider block for compatible endpoints."""
    doc: dict[str, Any] = {"$schema": SCHEMA_URL}
    if model.provider.type == "openai-compatible":
        info = model.info
        entry: dict[str, Any] = {"name": (info.display_name if info else None) or model.model.model}
        if info and info.context_window is not None and info.max_output_tokens is not None:
            entry["limit"] = {"context": info.context_window, "output": info.max_output_tokens}
        doc["provider"] = {
            model.provider.id: {
                "npm": "@ai-sdk/openai-compatible",
                "name": model.provider.display_name or model.provider.id,
                "options": {
                    "baseURL": model.base_url,
                    "apiKey": f"{{env:{model.provider.api_key_env}}}",
                },
                "models": {model.model.model: entry},
            }
        }
    doc["permissions"] = PERMISSIONS
    return json.dumps(doc, indent=2) + "\n"


class OpenCodeAdapter:
    name: ClassVar[str] = "opencode"
    display_name: ClassVar[str] = "OpenCode"
    executable: ClassVar[str] = "opencode"
    npm_package: ClassVar[str] = "@opencode/cli"

    def validate_reasoning(self, reasoning: str | None) -> None:
        # OpenCode "variants" are model-defined, so only the generic structural check applies.
        return None

    def build_invocation(self, model: ResolvedModel) -> HarnessInvocation:
        ref = f"{model.provider.id}/{model.model.model}"
        if model.model.reasoning is not None:
            ref += f"#{model.model.reasoning}"  # OpenCode's variant syntax
        argv = [
            CONTAINER_EXECUTABLE, "run", "--standalone", "--auto", "--agent", AGENT,
            "--model", ref, "--format", "json", PROMPT_PLACEHOLDER,
        ]  # fmt: skip
        return HarnessInvocation(
            argv=argv,
            env={"OPENCODE_CONFIG": CONFIG_PATH},
            secret_env=[model.provider.api_key_env],
            config_files=[
                GeneratedFile(
                    container_path=CONFIG_PATH,
                    content=render_config(model),
                    description="OpenCode benchmark policy and provider definition",
                )
            ],
            output=OutputExpectation(
                trace_format=self.display_name, stderr_file="opencode.stderr.log"
            ),
        )

    def provisioning(
        self, executable_path: str, host: HostEnvironment, staged_files: dict[str, Path]
    ) -> list[Provision]:
        """Frozen runners: the symlink-resolved opencode binary -> /usr/local/bin/opencode (ro)
        and the policy/provider config -> /etc/opencode/opencode.json (ro bind mount)."""
        real = host.realpath(executable_path)
        if not host.is_file(real):
            raise HarnessError(
                f"OpenCode executable {executable_path} resolves to {real}, which is not a file"
            )
        out = [
            Provision(
                kind="bind-mount", source=real,
                destination=CONTAINER_EXECUTABLE, read_only=True, purpose="opencode binary",
            )
        ]  # fmt: skip
        for container_path, staged in sorted(staged_files.items()):
            out.append(
                Provision(
                    kind="bind-mount", source=str(staged), destination=container_path,
                    read_only=True, purpose="generated OpenCode configuration",
                )
            )  # fmt: skip
        return out
