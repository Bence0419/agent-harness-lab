"""Pi adapter: reproduces the frozen SWE-bench runners' Pi treatment.

Source of truth: scripts/run-swebench-task{,-astra,-bme-qwen}.sh. The argv order and flag
set are the research treatment (print/json mode, no session, no extensions, skills, prompt
templates or context files, no approval, tools read,bash,edit,write). ``--thinking`` is
included only when a reasoning level is requested (the frozen Qwen runs omitted it).

Container layout of the frozen runners (a Phase 6 provisioning contract): the project's
``node_modules`` mounted at /opt/pi-node_modules and the host Node at /opt/node24.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

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

CONTAINER_EXECUTABLE = "/opt/pi-node_modules/.bin/pi"
CONTAINER_PATH = "/opt/node24/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
MODELS_JSON_PATH = "/root/.pi/agent/models.json"
CONTAINER_NODE_ROOT = "/opt/node24"
CONTAINER_NODE_MODULES = "/opt/pi-node_modules"
TOOLS = "read,bash,edit,write"
# From `pi --help`: --thinking <level>
THINKING_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh", "max")

FIXED_FLAGS = [
    "--print", "--mode", "json", "--no-session", "--no-extensions", "--no-skills",
    "--no-prompt-templates", "--no-context-files", "--no-approve", "--tools", TOOLS,
]  # fmt: skip


def render_models_json(model: ResolvedModel) -> str:
    """Pi models.json for an OpenAI-compatible provider (structure of the frozen config).

    Pi requires cost fields; zeros mean "unpriced", exactly as in the frozen config, so cost
    metrics for such runs are not meaningful.
    """
    info = model.info
    if info is None or info.context_window is None or info.max_output_tokens is None:
        raise HarnessError(
            f"Pi needs context_window and max_output_tokens for {model.model.model!r}: add them "
            f"under providers.{model.provider.id}.models in the providers file"
        )
    entry = {
        "id": model.model.model,
        "name": info.display_name or model.model.model,
    }
    if info.reasoning_supported is not None:  # unknown capability: omit, let Pi use its default
        entry["reasoning"] = info.reasoning_supported
    entry |= {
        "input": ["text"],
        "contextWindow": info.context_window,
        "maxTokens": info.max_output_tokens,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
    }
    doc = {
        "providers": {
            model.provider.id: {
                "baseUrl": model.base_url,
                "api": "openai-completions",
                "apiKey": f"${model.provider.api_key_env}",
                "authHeader": True,
                "compat": {"supportsReasoningEffort": False, "supportsDeveloperRole": False},
                "models": [entry],
            }
        }
    }
    return json.dumps(doc, indent=2) + "\n"


def nearest_node_modules(real_executable: str) -> Path:
    """The ``node_modules`` directory that contains a resolved executable."""
    for parent in Path(real_executable).parents:
        if parent.name == "node_modules":
            return parent
    raise HarnessError(
        f"cannot determine the node_modules directory of {real_executable}; "
        "the harness must be installed through npm"
    )


class PiAdapter:
    name: ClassVar[str] = "pi"
    display_name: ClassVar[str] = "Pi"
    executable: ClassVar[str] = "pi"
    npm_package: ClassVar[str] = "@earendil-works/pi-coding-agent"

    def validate_reasoning(self, reasoning: str | None) -> None:
        if reasoning is not None and reasoning not in THINKING_LEVELS:
            raise HarnessError(
                f"Pi does not support reasoning level {reasoning!r}; "
                f"choose one of: {', '.join(THINKING_LEVELS)}"
            )

    def build_invocation(self, model: ResolvedModel) -> HarnessInvocation:
        self.validate_reasoning(model.model.reasoning)
        argv = [CONTAINER_EXECUTABLE, "--provider", model.provider.id, "--model", model.model.model]
        if model.model.reasoning is not None:
            argv += ["--thinking", model.model.reasoning]
        argv += FIXED_FLAGS + [PROMPT_PLACEHOLDER]

        files: list[GeneratedFile] = []
        if model.provider.type == "openai-compatible":
            files.append(
                GeneratedFile(
                    container_path=MODELS_JSON_PATH,
                    content=render_models_json(model),
                    description="Pi custom provider/model definition",
                )
            )
        return HarnessInvocation(
            argv=argv,
            env={"PATH": CONTAINER_PATH},
            secret_env=[model.provider.api_key_env],
            config_files=files,
            output=OutputExpectation(trace_format=self.display_name, stderr_file="pi.stderr.log"),
        )

    def provisioning(
        self, executable_path: str, host: HostEnvironment, staged_files: dict[str, Path]
    ) -> list[Provision]:
        """Frozen runners: host Node root -> /opt/node24 (ro) and the project's node_modules
        -> /opt/pi-node_modules (ro); a models.json was mounted at /tmp and `cp`'d to
        /root/.pi/agent/models.json (a writable copy), here a direct copy-in."""
        node = host.which("node")
        if node is None:
            raise HarnessError("Pi needs Node.js on the host (not found on PATH) to provision")
        node_root = Path(host.realpath(node)).parent.parent
        modules = nearest_node_modules(host.realpath(executable_path))
        if not host.is_file(f"{node_root}/bin/node"):
            raise HarnessError(f"Node.js root {node_root} does not contain bin/node")
        if not host.is_dir(str(modules)):
            raise HarnessError(f"Pi installation directory does not exist: {modules}")
        out = [
            Provision(
                kind="bind-mount", source=str(node_root), destination=CONTAINER_NODE_ROOT,
                read_only=True, purpose="host Node.js runtime",
            ),
            Provision(
                kind="bind-mount", source=str(modules), destination=CONTAINER_NODE_MODULES,
                read_only=True, purpose="project node_modules containing Pi",
            ),
        ]  # fmt: skip
        for container_path, staged in sorted(staged_files.items()):
            out.append(
                Provision(
                    kind="copy-in", source=str(staged), destination=container_path,
                    read_only=False, purpose="generated Pi configuration",
                )
            )  # fmt: skip
        return out
