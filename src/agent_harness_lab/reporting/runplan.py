from __future__ import annotations

import shlex

from rich.console import Console
from rich.markup import escape

from agent_harness_lab.runplan.models import PlanResult


def render_plan(result: PlanResult, console: Console) -> None:
    plan = result.plan

    def row(label: str, value: str) -> None:
        console.print(f"[bold]{label + ':':<24}[/]{escape(value)}")

    def heading(text: str) -> None:
        console.print(f"\n[bold underline]{text}[/]")

    console.print("[bold]Run plan (dry run: nothing was executed)[/]")
    heading("Task")
    row("Task", plan.task.instance_id)
    row("Benchmark", plan.benchmark)
    dirty = {True: " (DIRTY)", False: "", None: " (dirty state unknown)"}[
        plan.task.task_repository_dirty
    ]
    row("Task repo commit", f"{plan.task.task_repository_commit or 'unknown'}{dirty}")
    row("Base commit", plan.task.base_commit)
    row("Docker image", plan.task.docker_image)

    heading("Harness and model")
    h = plan.harness
    row("Harness", f"{h.display_name} ({h.discovery_source}: {h.executable_path})")
    pin = f"  [expected pin: {h.expected_version}]" if h.expected_version else ""
    row("Harness version", f"{h.version}{pin}")
    row("Model", f"{plan.model.provider}/{plan.model.model}")
    row("Reasoning", plan.model.reasoning or "provider default")
    known = {True: "known", False: "known NOT supported", None: "unknown"}
    row("Reasoning support", known[plan.model_capabilities.reasoning_supported])

    heading("Constraints")
    row("Timeout", "none" if plan.timeout_seconds is None else f"{plan.timeout_seconds} s")
    hosts = ", ".join(plan.network.allowed_hosts) or "any"
    row("Network policy", f"{plan.network.mode.value} (allowed hosts: {hosts})")

    heading("Required environment variables")
    for name in plan.required_secret_env:
        state = "set" if result.secret_env_present.get(name) else "NOT SET"
        row(name, state)

    heading("Provenance")
    ahl_dirty = {True: "dirty", False: "clean", None: "unknown"}[plan.ahl.dirty]
    row("AHL commit", plan.ahl.commit or "unknown")
    row("Dirty state", ahl_dirty)

    heading("Agent invocation")
    shown = " ".join(shlex.quote(a) for a in plan.invocation.argv)  # display only
    console.print(escape(shown), soft_wrap=True)
    console.print(
        f"[dim]prompt: {plan.prompt.delivered_chars} chars, sha256 "
        f"{plan.prompt.delivered_sha256[:16]}…; env: "
        f"{', '.join(f'{k}=…' for k in plan.invocation.env) or 'none'}[/]"
    )
    for f in plan.invocation.config_files:
        console.print(f"[dim]generated config: {escape(f.container_path)} ({f.description})[/]")
    console.print(
        f"[dim]stdout -> {plan.invocation.output.stdout_file}, "
        f"stderr -> {plan.invocation.output.stderr_file}[/]"
    )
