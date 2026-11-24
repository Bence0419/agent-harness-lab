"""Deterministic, outcome-independent candidate selection for first canary runs.

Rule (``RULE``): sort tasks by instance id; keep those that are UNSEEN (no run directory in the
result store and not a frozen research task); take the first ``count`` whose Docker image is
already present locally. If fewer than ``count`` exist, say so and list the next unseen tasks
whose image is NOT local, clearly flagged, rather than silently padding.

Inputs are ordinary Task metadata (id, repo, image) and local image availability only. Evaluator
files and run outcomes are never consulted.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from pydantic import BaseModel, Field

from agent_harness_lab.benchmarks.base import Task

RULE = (
    "lexicographically first UNSEEN tasks (no run directory in the result store, not a frozen "
    "research task) whose Docker image is already present locally; if fewer exist, the "
    "lexicographically next unseen tasks are listed separately, flagged as missing their image"
)


class Candidate(BaseModel):
    instance_id: str
    repo: str
    base_commit: str
    docker_image: str
    image_local: bool


class Selection(BaseModel):
    rule: str = RULE
    requested: int
    unseen_total: int
    ready: list[Candidate] = Field(default_factory=list)  # unseen AND image present locally
    needs_image: list[Candidate] = Field(default_factory=list)  # unseen, image NOT local

    @property
    def shortfall(self) -> int:
        return max(0, self.requested - len(self.ready))


def select_candidates(
    tasks: Iterable[Task],
    *,
    is_seen: Callable[[str], bool],
    image_local: Callable[[str], bool],
    count: int = 3,
) -> Selection:
    unseen = [t for t in sorted(tasks, key=lambda t: t.instance_id) if not is_seen(t.instance_id)]

    def cand(t: Task, local: bool) -> Candidate:
        return Candidate(
            instance_id=t.instance_id,
            repo=t.repo,
            base_commit=t.base_commit,
            docker_image=t.docker_image,
            image_local=local,
        )

    ready = [cand(t, True) for t in unseen if image_local(t.docker_image)][:count]
    result = Selection(requested=count, unseen_total=len(unseen), ready=ready)
    missing = count - len(ready)
    if missing > 0:
        ready_ids = {c.instance_id for c in ready}
        result.needs_image = [cand(t, False) for t in unseen if t.instance_id not in ready_ids][
            :missing
        ]
    return result
