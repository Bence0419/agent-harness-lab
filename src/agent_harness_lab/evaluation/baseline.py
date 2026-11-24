"""Verify the benchmark baseline inside the evaluator image.

The official evaluator rebuilds the task image, and a patch-apply failure (or any verdict) is
only attributable to the model if that image really contains the task's baseline: the expected
base commit, a clean worktree. This inspects the EXACT evaluator image (by id, never the
restored shared tag) in a short-lived, network-less container. It reads environment provenance
only: no gold patch, no hints, no model output, no evaluator outcome.
"""

from __future__ import annotations

from agent_harness_lab.evaluation.models import EvaluatorImageBaseline
from agent_harness_lab.execution.docker import DockerClient
from agent_harness_lab.execution.models import ContainerSpec, DockerError

REASON_MISMATCH = "evaluator_image_baseline_mismatch"
REASON_DIRTY = "evaluator_image_baseline_dirty"
REASON_UNVERIFIED = "evaluator_image_baseline_unverified"


def verify_baseline(
    docker: DockerClient,
    image_id: str | None,
    expected_base_commit: str,
    *,
    workdir: str,
    label: str,
) -> EvaluatorImageBaseline:
    def result(status, *, reason=None, error=None, **kw) -> EvaluatorImageBaseline:
        return EvaluatorImageBaseline(
            verification_status=status,
            image_id=image_id,
            expected_base_commit=expected_base_commit,
            reason=reason,
            error=error,
            **kw,
        )

    def unverified(error: str, **kw) -> EvaluatorImageBaseline:
        return result("unverified", reason=REASON_UNVERIFIED, error=error, **kw)

    if image_id is None:
        return unverified("no evaluator image id is known")

    name = f"ahl-baseline-{label}"[:120]
    spec = ContainerSpec(
        name=name,
        image=image_id,
        network="none",  # an inspection needs no network
        env={},
        pass_env=[],
        mounts=[],
        labels={"ahl.managed": "true", "ahl.purpose": "evaluator-baseline-check"},
    )
    git = ["git", "-c", f"safe.directory={workdir}", "-C", workdir]
    try:
        try:
            docker.create_container(spec)
            docker.start(name)
            if docker.exec(name, ["test", "-d", workdir]).returncode != 0:
                return unverified(
                    f"{workdir} does not exist in the evaluator image", testbed_exists=False
                )
            inside = docker.exec(name, [*git, "rev-parse", "--is-inside-work-tree"])
            if inside.returncode != 0 or inside.stdout.strip() != "true":
                return unverified(
                    f"{workdir} is not a git repository",
                    testbed_exists=True,
                    is_git_repository=False,
                )
            head = docker.exec(name, [*git, "rev-parse", "HEAD"])
            if head.returncode != 0 or not head.stdout.strip():
                return unverified(
                    "git rev-parse HEAD failed", testbed_exists=True, is_git_repository=True
                )
            # same mode handling as the frozen workspace preparation (core.fileMode=false)
            status = docker.exec(
                name, ["git", "-c", "core.fileMode=false", *git[1:], "status", "--porcelain"]
            )
            if status.returncode != 0:
                return unverified(
                    "git status failed",
                    testbed_exists=True,
                    is_git_repository=True,
                    actual_base_commit=head.stdout.strip(),
                )
        finally:
            try:
                docker.remove(name)
            except DockerError:
                pass  # a leftover inspection container is harmless and labelled
    except DockerError as exc:
        return unverified(f"could not inspect the evaluator image: {exc}")

    actual = head.stdout.strip()
    clean = status.stdout.strip() == ""
    common = dict(
        testbed_exists=True, is_git_repository=True, actual_base_commit=actual, worktree_clean=clean
    )
    if actual != expected_base_commit:
        return result(
            "mismatch",
            reason=REASON_MISMATCH,
            error=f"image /testbed is at {actual}, the task expects {expected_base_commit}",
            **common,
        )
    if not clean:
        return result(
            "dirty",
            reason=REASON_DIRTY,
            error="image /testbed has uncommitted changes at the expected commit",
            **common,
        )
    return result("verified", **common)
