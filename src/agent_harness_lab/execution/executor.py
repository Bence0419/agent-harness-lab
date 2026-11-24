"""Execute one RunPlan in an isolated task container and persist the evidence.

The executor knows Docker, the workspace contract and the artifact layout; it does NOT know how
Pi or OpenCode are invoked (that is in the plan's HarnessInvocation and the adapters).

Stages (each failure becomes a named ``failed_stage``): environment, prompt, artifacts, image,
provisioning, network, container, workspace, harness, patch. Cleanup always runs, and a cleanup
failure is recorded without replacing the original failure.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from agent_harness_lab.artifacts.hashing import sha256_file, sha256_text
from agent_harness_lab.artifacts.run_record import (
    METADATA_FILE,
    RUN_FILE,
    Hashes,
    RunDocument,
    compat_metadata,
)
from agent_harness_lab.artifacts.writer import write_json_atomic
from agent_harness_lab.execution.docker import DockerClient
from agent_harness_lab.execution.gate import GATE_ENV, real_execution_enabled
from agent_harness_lab.execution.models import (
    CleanupRecord,
    DockerError,
    ExecutionDisabled,
    ExecutionState,
    Provision,
    StageError,
)
from agent_harness_lab.execution.network import (
    NetworkResources,
    agent_container_spec,
    plan_network,
    start_network,
)
from agent_harness_lab.execution.workspace import capture_patch, prepare_workspace
from agent_harness_lab.harnesses import PROMPT_PLACEHOLDER, available_harnesses
from agent_harness_lab.harnesses.base import HarnessError
from agent_harness_lab.runplan.models import RunPlan

SWEBENCH_RAW_SUBDIR = "swebench"
GENERATED_DIRNAME = "generated"


@dataclass
class RunRequest:
    plan: RunPlan
    prompt: str  # exact delivered prompt (its sha256 must equal plan.prompt.delivered_sha256)
    # Maps staged host paths of generated config files to everything that must be mounted/copied.
    # Supplied by the caller (normally a harness adapter), so this layer stays harness-agnostic.
    provisioner: Callable[[dict[str, Path]], list[Provision]]
    raw_dir: Path  # results/raw (or any directory; new runs are created under swebench/)
    plan_warnings: list[str] = field(default_factory=list)


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def utc_now() -> datetime:
    return datetime.now(UTC)


def allocate_run_dir(
    raw_dir: Path, task_id: str, harness_name: str, now: datetime
) -> tuple[str, Path]:
    """Create a NEW run directory; never reuses or overwrites an existing one.

    Run id: ``<task>-<harness>-<UTC timestamp>`` (+ ``-N`` if that second is already taken).
    """
    task_dir = raw_dir / SWEBENCH_RAW_SUBDIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    base = f"{task_id}-{harness_name}-{now.astimezone(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    for n in range(1, 1000):
        run_id = base if n == 1 else f"{base}-{n}"
        try:
            (task_dir / run_id).mkdir()
        except FileExistsError:
            continue
        return run_id, task_dir / run_id
    raise RuntimeError(f"could not allocate a unique run directory for {base}")


@contextmanager
def _stage(name: str) -> Iterator[None]:
    """Turn infrastructure errors in a stage into a named StageError."""
    try:
        yield
    except StageError:
        raise
    except (DockerError, OSError, HarnessError) as exc:
        raise StageError(name, str(exc)) from exc


def _safe_name(run_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "-", run_id)


@dataclass
class _Created:
    """Resources actually created, so cleanup removes exactly those."""

    container: str | None = None


def execute_run(
    request: RunRequest,
    *,
    docker: DockerClient,
    now: Callable[[], datetime] = utc_now,
    sleep: Callable[[float], None] = time.sleep,
    environ: Mapping[str, str] | None = None,
) -> RunDocument:
    plan = request.plan
    env = environ if environ is not None else {}
    if plan.harness.name in available_harnesses() and not real_execution_enabled(env):
        raise ExecutionDisabled(
            f"real execution of {plan.harness.name!r} is disabled: the temporary safety gate "
            f"{GATE_ENV}=1 is not set"
        )

    secret_values = [v for n in plan.required_secret_env if (v := env.get(n))]

    def redact(text: str) -> str:
        for value in secret_values:
            text = text.replace(value, "***")
        return text

    started = now()
    run_id, run_dir = allocate_run_dir(
        request.raw_dir, plan.task.instance_id, plan.harness.name, started
    )
    doc = RunDocument(
        run_id=run_id,
        state=ExecutionState.PREPARED,
        started_at=_iso(started),
        plan=plan,
        hashes=Hashes(prompt_sha256=sha256_text(request.prompt)),
        warnings=list(request.plan_warnings),
    )

    def persist() -> None:
        write_json_atomic(run_dir / RUN_FILE, doc.model_dump(mode="json"))

    persist()  # a valid run.json exists from the very start

    safe = _safe_name(run_id)
    labels = {"ahl.managed": "true", "ahl.run-id": run_id}
    created = _Created()
    resources: NetworkResources = plan_network(
        plan.network, names=(f"ahl-net-{safe}", f"ahl-proxy-{safe}")
    )
    generated_dir = run_dir / GENERATED_DIRNAME
    out_spec = plan.invocation.output
    interrupted: BaseException | None = None

    try:
        with _stage("environment"):
            missing = [n for n in plan.required_secret_env if not env.get(n)]
            if missing:
                raise StageError(
                    "environment",
                    f"required environment variable(s) not set: {', '.join(missing)}",
                )

        with _stage("prompt"):
            if sha256_text(request.prompt) != plan.prompt.delivered_sha256:
                raise StageError("prompt", "prompt does not match the plan's recorded sha256")
            (run_dir / "prompt.txt").write_text(request.prompt, encoding="utf-8")

        staged: dict[str, Path] = {}
        with _stage("artifacts"):
            generated_dir.mkdir()
            for i, gf in enumerate(plan.invocation.config_files):
                path = generated_dir / f"{i:02d}-{Path(gf.container_path).name}"
                path.write_text(gf.content, encoding="utf-8")
                staged[gf.container_path] = path
                doc.hashes.generated_config_sha256[gf.container_path] = sha256_text(gf.content)

        with _stage("image"):
            try:
                doc.image = docker.image_inspect(plan.task.docker_image)
            except DockerError as exc:
                raise StageError("image", str(exc)) from exc

        with _stage("provisioning"):
            hashes = doc.hashes.generated_config_sha256
            doc.provisioning = [
                p.model_copy(update={"sha256": hashes.get(p.destination, p.sha256)})
                for p in request.provisioner(staged)
            ]

        with _stage("network"):
            start_network(docker, resources, labels=labels, staging_dir=generated_dir, sleep=sleep)
        doc.network = resources.record

        with _stage("container"):
            spec = agent_container_spec(
                name=f"ahl-agent-{safe}",
                image=plan.task.docker_image,
                resources=resources,
                env=plan.invocation.env,
                pass_env=plan.invocation.secret_env,
                mounts=[p for p in doc.provisioning if p.kind == "bind-mount"],
                labels=labels,
            )
            doc.container_name = spec.name
            docker.create_container(spec)
            created.container = spec.name
            docker.start(spec.name)
            doc.container_id = docker.inspect_container(spec.name).get("Id")

        with _stage("provisioning"):
            for p in (p for p in doc.provisioning if p.kind == "copy-in"):
                parent = p.destination.rsplit("/", 1)[0] or "/"
                made = docker.exec(spec.name, ["mkdir", "-p", parent])
                if made.returncode != 0:
                    raise StageError("provisioning", f"cannot create {parent} in the container")
                docker.copy_in(Path(p.source), spec.name, p.destination)

        with _stage("workspace"):
            prepare_workspace(docker, spec.name, plan.task.base_commit, plan.workdir)

        # ---- run the harness ------------------------------------------------------------
        argv = [prompt_or(a, request.prompt) for a in plan.invocation.argv]
        agent_start = now()
        doc.state = ExecutionState.RUNNING
        doc.agent_started_at = _iso(agent_start)
        persist()
        with _stage("harness"):
            result = docker.exec(
                spec.name,
                argv,
                workdir=plan.workdir,
                timeout=plan.timeout_seconds,  # None -> no timeout at all
                stdout_path=run_dir / out_spec.stdout_file,
                stderr_path=run_dir / out_spec.stderr_file,
            )
        agent_end = now()
        doc.agent_finished_at = _iso(agent_end)
        doc.duration_seconds = round((agent_end - agent_start).total_seconds(), 3)
        doc.exit_code = result.returncode
        if result.timed_out:
            doc.state = ExecutionState.TIMED_OUT
            persist()  # the timeout outcome is on disk before anything else is attempted
            _stop_agent_after_timeout(docker, spec.name, doc, redact)
        elif result.returncode != 0:
            doc.state = ExecutionState.HARNESS_FAILED
        else:
            doc.state = ExecutionState.COMPLETED

        # Evidence is captured whatever the harness did (as in the frozen runners).
        try:
            with _stage("patch"):
                capture_patch(docker, spec.name, plan.task.base_commit, plan.workdir, run_dir)
        except StageError as exc:
            if doc.state is ExecutionState.COMPLETED:
                doc.state, doc.failed_stage, doc.error = (
                    ExecutionState.FAILED,
                    exc.stage,
                    redact(exc.message),
                )
            else:
                doc.warnings.append(f"patch capture failed: {redact(exc.message)}")

    except StageError as exc:
        doc.state, doc.failed_stage, doc.error = (
            ExecutionState.FAILED,
            exc.stage,
            redact(exc.message),
        )
    except KeyboardInterrupt as exc:
        doc.state, doc.failed_stage, doc.error = (
            ExecutionState.FAILED,
            "interrupted",
            "interrupted by user",
        )
        interrupted = exc
    except Exception as exc:  # unexpected: still leave a valid, explanatory run.json
        doc.state, doc.failed_stage = ExecutionState.FAILED, "internal"
        doc.error = redact(f"{type(exc).__name__}: {exc}")
    finally:
        doc.cleanup = _cleanup(docker, resources, created, run_dir, redact)
        _finalize(doc, run_dir, now)

    if interrupted is not None:
        raise interrupted
    return doc


def _stop_agent_after_timeout(
    docker: DockerClient, container: str, doc: RunDocument, redact: Callable[[str], str]
) -> None:
    """Make sure the timed-out agent is really dead, touching ONLY this run's container.

    Terminating the host-side ``docker exec`` client does not stop the in-container process, so
    the container is killed. Its filesystem survives a kill, so it is then restarted: the idle
    ``sleep`` entrypoint gives a quiescent workspace for patch capture with no agent still
    editing files. Both steps are best-effort; failures are recorded as warnings and never
    change the ``timed_out`` outcome.
    """
    try:
        docker.kill(container)
    except Exception as exc:
        doc.timeout_action = "kill-failed"
        doc.warnings.append(redact(f"container kill after timeout failed: {exc}"))
        return
    doc.timeout_action = "killed"
    try:
        docker.start(container)
    except Exception as exc:
        doc.warnings.append(
            redact(f"container restart for evidence capture failed after timeout kill: {exc}")
        )


def prompt_or(arg: str, prompt: str) -> str:
    """Substitute the prompt placeholder (a whole argv element); other elements are untouched."""
    return prompt if arg == PROMPT_PLACEHOLDER else arg


def _cleanup(
    docker: DockerClient,
    resources: NetworkResources,
    created: _Created,
    run_dir: Path,
    redact: Callable[[str], str],
) -> CleanupRecord:
    """Best effort, every step independent; failures are recorded, never raised."""
    errors: list[str] = []
    attempted = False

    if resources.proxy_name:
        attempted = True
        try:
            (run_dir / "proxy.log").write_text(docker.logs(resources.proxy_name), encoding="utf-8")
        except Exception as exc:
            errors.append(redact(f"proxy log collection failed: {exc}"))

    names = [n for n in (created.container, resources.proxy_name) if n]
    if names:
        attempted = True
        try:
            docker.remove(*names)
        except Exception as exc:
            errors.append(redact(f"container removal failed: {exc}"))
    if resources.network_name:
        attempted = True
        try:
            docker.network_remove(resources.network_name)
        except Exception as exc:
            errors.append(redact(f"network removal failed: {exc}"))

    if not attempted:
        return CleanupRecord(status="not-needed")
    return CleanupRecord(status="failed" if errors else "ok", errors=errors)


def _finalize(doc: RunDocument, run_dir: Path, now: Callable[[], datetime]) -> None:
    """Hash finalized artifacts, then write run.json and the compatibility metadata.json."""
    out = doc.plan.invocation.output
    names = [
        "prompt.txt",
        out.stdout_file,
        out.stderr_file,
        "model.patch",
        "git-status.txt",
        "proxy.log",
    ]
    doc.hashes.artifacts = {n: sha256_file(run_dir / n) for n in names if (run_dir / n).is_file()}
    doc.hashes.patch_sha256 = doc.hashes.artifacts.get("model.patch")
    doc.hashes.trace_sha256 = doc.hashes.artifacts.get(out.stdout_file)
    doc.finished_at = _iso(now())
    write_json_atomic(run_dir / RUN_FILE, doc.model_dump(mode="json"))
    write_json_atomic(run_dir / METADATA_FILE, compat_metadata(doc))
