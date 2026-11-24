"""Has a task ever been run? Used to keep canaries away from already-seen tasks.

"Seen" means: any run directory exists for the task in the result store (complete or not), or
the task belongs to a frozen research manifest. Only run metadata is read: never evaluator
output or task evaluator files, and never the outcome of a run.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from agent_harness_lab.artifacts.discovery import SWEBENCH_SUBDIR
from agent_harness_lab.artifacts.legacy import LegacyFormatError, detect_format
from agent_harness_lab.artifacts.run_record import RUN_FILE

FROZEN_MANIFESTS = (
    "experiments/swebench/swebench-pilot-5-v1.json",
    "configs/challenge-set-v1.json",
)


class TaskHistory(BaseModel):
    instance_id: str
    seen: bool
    run_directories: int
    frozen_research_task: bool
    harnesses: list[str] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    purposes: list[str] = Field(default_factory=list)


def frozen_task_ids(project_root: Path | None) -> set[str]:
    """Tasks of the frozen research studies, from the committed manifests (best effort)."""
    if project_root is None:
        return set()
    ids: set[str] = set()
    for rel in FROZEN_MANIFESTS:
        path = project_root / rel
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        ids.update(i["instanceId"] for i in doc.get("instances", []) if "instanceId" in i)
        for group in (doc.get("ecosystems") or {}).values():
            ids.update(group)
    return ids


def _describe(run_dir: Path) -> tuple[str, str | None, str | None] | None:
    """(harness, model, purpose) from run METADATA only. Outcomes (evaluation.json) are never
    opened: whether a task was solved must not influence canary selection."""
    try:
        if (run_dir / RUN_FILE).is_file():
            plan = json.loads((run_dir / RUN_FILE).read_text(encoding="utf-8"))["plan"]
            return plan["harness"]["display_name"], plan["model"]["model"], plan.get("purpose")
        meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        harness = "OpenCode" if detect_format(meta).startswith("opencode") else "Pi"
        model = meta.get("model")
        return harness, model if isinstance(model, str) else None, None
    except (OSError, ValueError, KeyError, TypeError, LegacyFormatError):
        return None


def task_history(
    raw_dir: Path, instance_id: str, frozen: set[str] | frozenset[str] = frozenset()
) -> TaskHistory:
    task_dir = raw_dir / SWEBENCH_SUBDIR / instance_id
    run_dirs = sorted(p for p in task_dir.iterdir() if p.is_dir()) if task_dir.is_dir() else []
    harnesses: set[str] = set()
    models: set[str] = set()
    purposes: set[str] = set()
    for d in run_dirs:
        described = _describe(d)
        if described is None:
            continue  # incomplete/unreadable directory: still counts as history
        harness, model, purpose = described
        harnesses.add(harness)
        if model:
            models.add(model)
        if purpose:
            purposes.add(purpose)
    is_frozen = instance_id in frozen
    return TaskHistory(
        instance_id=instance_id,
        seen=bool(run_dirs) or is_frozen,
        run_directories=len(run_dirs),
        frozen_research_task=is_frozen,
        harnesses=sorted(harnesses),
        models=sorted(models),
        purposes=sorted(purposes),
    )
