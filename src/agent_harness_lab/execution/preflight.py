"""Static pre-flight validation for a resolved RunPlan.

Everything here is read-only: file existence checks, environment-variable PRESENCE checks and
``docker image inspect``. It never creates a container, network or proxy, never runs the
harness and never calls a model API. Secret values are never read into the report.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, computed_field

from agent_harness_lab.artifacts.hashing import sha256_text
from agent_harness_lab.execution.docker import DockerClient
from agent_harness_lab.execution.gate import GATE_ENV, real_execution_enabled
from agent_harness_lab.execution.models import DockerError, ImageRecord, Provision
from agent_harness_lab.execution.network import SQUID_IMAGE, render_squid_policy
from agent_harness_lab.harnesses import available_harnesses
from agent_harness_lab.harnesses.base import HarnessError
from agent_harness_lab.runplan.models import NetworkMode, RunPlan

STAGING_PLACEHOLDER = "<generated at run time>"


class CheckStatus(StrEnum):
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"


class PreflightCheck(BaseModel):
    name: str
    status: CheckStatus
    detail: str = ""


class GeneratedFileSummary(BaseModel):
    container_path: str
    sha256: str
    bytes: int


class PreflightReport(BaseModel):
    checks: list[PreflightCheck] = Field(default_factory=list)
    provisioning: list[Provision] = Field(default_factory=list)  # what WOULD be mounted/copied
    generated_files: list[GeneratedFileSummary] = Field(default_factory=list)
    image: ImageRecord | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ready(self) -> bool:
        return not any(c.status is CheckStatus.ERROR for c in self.checks)

    def add(self, name: str, status: CheckStatus, detail: str = "") -> None:
        self.checks.append(PreflightCheck(name=name, status=status, detail=detail))


def run_preflight(
    plan: RunPlan,
    *,
    provisioner: Callable[[dict[str, Path]], list[Provision]],
    docker: DockerClient,
    path_exists: Callable[[str], bool],
    environ: Mapping[str, str],
    plan_warnings: list[str] | None = None,
) -> PreflightReport:
    report = PreflightReport()
    ok, warn, err = CheckStatus.OK, CheckStatus.WARNING, CheckStatus.ERROR

    # -- gate --
    if plan.harness.name in available_harnesses():
        if real_execution_enabled(environ):
            report.add("real-execution gate", ok, f"{GATE_ENV}=1")
        else:
            report.add("real-execution gate", err, f"{GATE_ENV}=1 is not set (required to execute)")
    else:
        report.add("real-execution gate", ok, "not required for this harness")

    # -- task and provenance (already loaded by the plan builder) --
    t = plan.task
    report.add("task", ok, f"{t.instance_id} loaded ({t.repo} @ {t.base_commit[:12]})")
    if t.task_repository_commit is None:
        report.add("task repository provenance", err, "task repository commit is unavailable")
    elif t.task_repository_dirty:
        report.add("task repository provenance", warn, f"{t.task_repository_commit} (DIRTY)")
    else:
        report.add("task repository provenance", ok, t.task_repository_commit)
    if plan.ahl.dirty:
        report.add("AHL working tree", warn, "uncommitted changes (affects reproducibility)")

    # -- harness --
    h = plan.harness
    report.add(
        "harness executable",
        ok,
        f"{h.display_name} {h.version} ({h.discovery_source}: {h.executable_path})",
    )
    if h.expected_version and h.expected_version != h.version:
        report.add(
            "harness version pin",
            warn,
            f"{h.version} differs from repository pin {h.expected_version}",
        )
    elif h.expected_version:
        report.add("harness version pin", ok, f"matches repository pin {h.expected_version}")

    # -- generated configuration (rendered when the plan was built) --
    staged: dict[str, Path] = {}
    for i, f in enumerate(plan.invocation.config_files):
        staged[f.container_path] = (
            Path(STAGING_PLACEHOLDER) / f"{i:02d}-{Path(f.container_path).name}"
        )
        report.generated_files.append(
            GeneratedFileSummary(
                container_path=f.container_path,
                sha256=sha256_text(f.content),
                bytes=len(f.content.encode("utf-8")),
            )
        )
        if f.container_path.endswith(".json"):
            try:
                json.loads(f.content)
                report.add("generated configuration", ok, f"{f.container_path} is valid JSON")
            except ValueError:
                report.add("generated configuration", err, f"{f.container_path} is not valid JSON")
        else:
            report.add("generated configuration", ok, f.container_path)

    # -- provisioning sources --
    try:
        provisions = provisioner(staged)
    except HarnessError as exc:
        report.add("provisioning", err, str(exc))
        provisions = []
    report.provisioning = provisions
    generated_sources = {str(p) for p in staged.values()}
    for p in provisions:
        if p.source in generated_sources:
            report.add("provisioning source", ok, f"{p.destination}: generated at run time")
        elif path_exists(p.source):
            report.add("provisioning source", ok, f"{p.source} -> {p.destination}")
        else:
            report.add("provisioning source", err, f"missing: {p.source} (for {p.destination})")

    # -- secrets (presence only) --
    for name in plan.required_secret_env:
        if environ.get(name):
            report.add("secret", ok, f"{name} is set")
        else:
            report.add("secret", err, f"{name} is not set")

    # -- network --
    net = plan.network
    if plan.purpose == "canary" and net.mode is not NetworkMode.RESTRICTED_PROVIDER:
        report.add("network plan", err, "canary runs require restricted-provider networking")
    elif net.mode is NetworkMode.RESTRICTED_PROVIDER:
        if not net.allowed_hosts:
            report.add("network plan", err, "restricted-provider has no allowed host")
        else:
            render_squid_policy(net.allowed_hosts)  # raises only if hosts are empty
            report.add(
                "network plan", ok, f"restricted-provider, allowed: {', '.join(net.allowed_hosts)}"
            )
    else:
        report.add("network plan", warn, "unrestricted")

    # -- Docker images (read-only inspection) --
    try:
        report.image = docker.image_inspect(t.docker_image)
        report.add("task image", ok, f"{t.docker_image} ({(report.image.image_id or '')[:19]})")
    except DockerError as exc:
        report.add("task image", err, str(exc))
    if net.mode is NetworkMode.RESTRICTED_PROVIDER:
        try:
            docker.image_inspect(SQUID_IMAGE)
            report.add("proxy image", ok, "pinned Squid image present")
        except DockerError as exc:
            report.add("proxy image", err, str(exc))

    return report
