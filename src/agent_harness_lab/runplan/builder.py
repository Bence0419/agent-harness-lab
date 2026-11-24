"""Resolve user intent into a RunPlan. Pure planning: no Docker, no agent, no network."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from agent_harness_lab import __version__
from agent_harness_lab.benchmarks import BenchmarkAdapter, BenchmarkError
from agent_harness_lab.benchmarks.git import GitProbe
from agent_harness_lab.harnesses import HarnessError, get_harness
from agent_harness_lab.harnesses.discovery import (
    DiscoveryProbe,
    discover_executable,
    parse_version,
    pinned_version,
)
from agent_harness_lab.providers import (
    ProviderError,
    ProviderSpec,
    reasoning_capability,
    resolve_model,
    validate_reasoning,
)
from agent_harness_lab.runplan.models import (
    DEFAULT_TIMEOUT_SECONDS,
    AhlRef,
    ExperimentRef,
    HarnessRef,
    ModelCapabilities,
    NetworkMode,
    NetworkPlan,
    PlanResult,
    PromptRef,
    ProviderRef,
    Purpose,
    RunPlan,
    TaskRef,
)
from agent_harness_lab.runplan.prompt import PREAMBLE, build_agent_prompt, prompt_sha256


class PlanError(BenchmarkError):
    """The run cannot be planned (reported without a traceback)."""


class PlanProbe(DiscoveryProbe, Protocol):
    def run(self, argv: list[str]): ...
    def getenv(self, name: str) -> str | None: ...


class _EnvView:
    """Read-only mapping view over the probe's environment (no values are retained)."""

    def __init__(self, probe: PlanProbe) -> None:
        self._probe = probe

    def get(self, name: str, default: str | None = None) -> str | None:
        value = self._probe.getenv(name)
        return default if value is None else value


def build_run_plan(
    *,
    benchmark: BenchmarkAdapter,
    task_id: str,
    harness_name: str,
    model_ref: str,
    reasoning: str | None,
    timeout_seconds: int | None,
    network_mode: NetworkMode,
    providers: Mapping[str, ProviderSpec],
    probe: PlanProbe,
    git: GitProbe,
    purpose: Purpose = "canary",
    experiment: ExperimentRef | None = None,
) -> PlanResult:
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise PlanError(
            f"timeout must be a positive number of seconds (or None for no timeout), "
            f"got {timeout_seconds}"
        )
    try:
        harness = get_harness(harness_name)
        validate_reasoning(reasoning)
        harness.validate_reasoning(reasoning)
        model = resolve_model(model_ref, reasoning, providers, _EnvView(probe))
    except (ProviderError, HarnessError) as exc:
        raise PlanError(str(exc)) from exc

    # ---- reasoning: requested level vs known capability -------------------------------
    warnings: list[str] = []
    capability = reasoning_capability(model)
    label = f"{model.provider.id}/{model.model.model}"
    if reasoning is not None and capability is False:
        raise PlanError(
            f"{label} is configured as not supporting reasoning selection, but --reasoning "
            f"{reasoning!r} was requested; remove it or correct reasoning_supported"
        )
    if reasoning is not None and capability is None:
        warnings.append(
            f"reasoning support for {label} is unknown: the requested level {reasoning!r} is "
            "passed to the harness unverified (it may be ignored or rejected)"
        )

    task = benchmark.get_task(task_id)
    info = benchmark.repository_info()

    # ---- harness discovery (PATH, then project-local) ------------------------------
    found = discover_executable(probe, harness.executable)
    if found is None:
        raise PlanError(
            f"{harness.display_name} executable '{harness.executable}' not found "
            "(searched PATH and, in a source checkout, node_modules/.bin; install it with npm)"
        )
    result = probe.run([found.path, "--version"])
    version = None
    if result is not None and result.returncode == 0:
        version = parse_version(result.output)
    if version is None:
        raise PlanError(f"could not detect the version of {found.path} (`--version` failed)")
    expected = pinned_version(probe, harness.npm_package)
    if expected is not None and expected != version:
        warnings.append(
            f"{harness.display_name} version {version} ({found.source}) differs from the "
            f"repository pin {expected}: results may not be comparable with the frozen study"
        )

    # ---- invocation, secrets, network ---------------------------------------------
    try:
        invocation = harness.build_invocation(model)
    except HarnessError as exc:
        raise PlanError(str(exc)) from exc
    secret_names = list(invocation.secret_env)
    present = {name: bool(probe.getenv(name)) for name in secret_names}
    for name, ok in present.items():
        if not ok:
            warnings.append(f"required environment variable {name} is not set (run would fail)")

    if timeout_seconds is None:
        warnings.append("no execution timeout: a hung agent would run until stopped")

    allowed = [model.endpoint_host] if network_mode is NetworkMode.RESTRICTED_PROVIDER else []
    if network_mode is NetworkMode.UNRESTRICTED:
        warnings.append("network is unrestricted: the agent could reach any host")

    # ---- provenance ---------------------------------------------------------------
    if info.commit is None:
        warnings.append("task repository commit could not be read")
    if info.dirty:
        warnings.append("task repository has uncommitted changes (affects reproducibility)")
    elif info.dirty is None:
        warnings.append("task repository dirty state could not be determined")

    root = probe.project_root()
    ahl_commit = git.head_commit(Path(root)) if root else None
    ahl_dirty = git.is_dirty(Path(root)) if root else None
    if ahl_commit is None:
        warnings.append(
            "AHL repository commit unknown (not a git source checkout): only the AHL version "
            f"{__version__} is recorded"
        )
    if ahl_dirty:
        warnings.append("AHL working tree has uncommitted changes (affects reproducibility)")

    prompt = build_agent_prompt(task.problem_statement)
    plan = RunPlan(
        purpose=purpose,
        benchmark=benchmark.name,
        task=TaskRef(
            instance_id=task.instance_id,
            repo=task.repo,
            base_commit=task.base_commit,
            docker_image=task.docker_image,
            task_repository_commit=info.commit,
            task_repository_dirty=info.dirty,
            split=task.split,
            version=task.version,
        ),
        prompt=PromptRef(
            preamble=PREAMBLE, delivered_sha256=prompt_sha256(prompt), delivered_chars=len(prompt)
        ),
        workdir=benchmark.container_workdir,
        harness=HarnessRef(
            name=harness.name,
            display_name=harness.display_name,
            executable_path=found.path,
            discovery_source=found.source,
            version=version,
            expected_version=expected,
        ),
        invocation=invocation,
        provider=ProviderRef(
            id=model.provider.id,
            type=model.provider.type,
            display_name=model.provider.display_name,
            base_url=model.base_url,
            base_url_env=model.provider.base_url_env,
            api_key_env=model.provider.api_key_env,
        ),
        model=model.model,
        model_capabilities=ModelCapabilities(reasoning_supported=capability),
        timeout_seconds=timeout_seconds,
        network=NetworkPlan(mode=network_mode, allowed_hosts=allowed),
        required_secret_env=secret_names,
        ahl=AhlRef(commit=ahl_commit, dirty=ahl_dirty, version=__version__),
        experiment=experiment,
    )
    return PlanResult(plan=plan, warnings=warnings, secret_env_present=present)


__all__ = ["DEFAULT_TIMEOUT_SECONDS", "PlanError", "PlanProbe", "build_run_plan"]
