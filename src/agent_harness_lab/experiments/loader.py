"""Read and validate an experiment file; resolve task-set files; compute identity."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import ValidationError

from agent_harness_lab.experiments.errors import ExperimentError
from agent_harness_lab.experiments.models import (
    EXPERIMENT_SCHEMA,
    ExperimentDefinition,
    NormalizedExperiment,
    TaskSetFile,
    TaskSetRecord,
    sanitize_name,
)


@dataclass(frozen=True)
class LoadedExperiment:
    source: Path
    raw_bytes: bytes  # copied byte-for-byte into the experiment directory
    definition: ExperimentDefinition
    normalized: NormalizedExperiment

    @property
    def sha256(self) -> str:
        return self.normalized.sha256


def _format_errors(exc: ValidationError) -> str:
    lines = []
    for e in exc.errors(include_input=False, include_url=False):
        where = ".".join(str(p) for p in e["loc"]) or "(root)"
        lines.append(f"{where}: {e['msg']}")
    return "; ".join(lines)


def read_task_set(path: Path) -> tuple[list[str], str]:
    """Task ids from a task-set file and the sha256 of its bytes.

    Accepted shapes (JSON or YAML): ``{"tasks": [...]}`` or the ecosystem layout of
    configs/challenge-set-v1.json (``{"ecosystems": {name: [...]}}``). Other keys (for example
    ``pilotTasksExcluded``) are ignored: they document the set and are not part of it.
    """
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ExperimentError(f"cannot read task set {path}: {exc.strerror}") from exc
    try:
        data = yaml.safe_load(raw.decode("utf-8"))  # YAML is a superset of JSON
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        raise ExperimentError(f"task set {path} is not valid JSON/YAML") from exc
    if isinstance(data, dict) and isinstance(data.get("tasks"), list):
        ids = data["tasks"]
    elif isinstance(data, dict) and isinstance(data.get("ecosystems"), dict):
        ids = [t for key in sorted(data["ecosystems"]) for t in _as_list(data["ecosystems"][key])]
    else:
        raise ExperimentError(f"task set {path} has neither 'tasks' nor 'ecosystems'")
    if not ids or not all(isinstance(t, str) and t for t in ids):
        raise ExperimentError(f"task set {path} must list one or more task ids (strings)")
    return list(ids), hashlib.sha256(raw).hexdigest()


def _as_list(value: object) -> list:
    return value if isinstance(value, list) else []


def parse_experiment(raw: bytes, *, base_dir: Path, source: Path) -> LoadedExperiment:
    try:
        data = yaml.safe_load(raw.decode("utf-8"))
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        problem = getattr(exc, "problem", None) or "invalid YAML"
        raise ExperimentError(f"{source.name} is not valid YAML: {problem}") from exc
    if not isinstance(data, dict):
        raise ExperimentError(f"{source.name} must contain a mapping at the top level")
    if data.get("schema_version") != EXPERIMENT_SCHEMA:
        raise ExperimentError(
            f"unsupported schema_version {data.get('schema_version')!r}; "
            f"this version of ahl reads {EXPERIMENT_SCHEMA!r}"
        )
    try:
        definition = ExperimentDefinition.model_validate(data)
    except ValidationError as exc:
        raise ExperimentError(f"invalid experiment {source.name}: {_format_errors(exc)}") from exc

    task_set: TaskSetRecord | None = None
    if isinstance(definition.tasks, TaskSetFile):
        file = Path(definition.tasks.file)
        ids, digest = read_task_set(file if file.is_absolute() else base_dir / file)
        dupes = sorted({t for t in ids if ids.count(t) > 1})
        if dupes:
            raise ExperimentError(f"duplicate task(s) in the task set: {', '.join(dupes)}")
        task_set = TaskSetRecord(file=definition.tasks.file, sha256=digest)
    else:
        ids = list(definition.tasks)

    normalized = NormalizedExperiment(
        experiment=definition.experiment,
        benchmark=definition.benchmark,
        tasks=sorted(ids),
        task_set=task_set,
        harnesses=sorted(definition.harnesses),
        models=sorted(definition.models, key=lambda m: (m.provider, m.model, m.reasoning or "")),
        repetitions=definition.repetitions,
        execution=definition.execution,
        evaluation=definition.evaluation,
    )
    return LoadedExperiment(source, raw, definition, normalized)


def load_experiment(path: Path) -> LoadedExperiment:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ExperimentError(f"cannot read experiment file {path}: {exc.strerror}") from exc
    return parse_experiment(raw, base_dir=path.resolve().parent, source=path)


def make_experiment_id(name: str, started: datetime, definition_sha256: str) -> str:
    stamp = started.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{sanitize_name(name)}-{stamp}-{definition_sha256[:8]}"


def dump_json(data: object) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)
