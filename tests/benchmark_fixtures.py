"""Synthetic swe-bench-multilingual-tasks checkout. Evaluator files carry sentinels."""

from __future__ import annotations

from pathlib import Path

import yaml

COMMIT = "6e08cbcd763c869abd7aae4bcb9e1b3991180777"

HIDDEN_FILES = {
    "gold.patch": "SENTINEL_GOLD_PATCH",
    "hints.md": "SENTINEL_HINTS",
    "test.patch": "SENTINEL_TEST_PATCH",
    "tests.json": '{"fail_to_pass": ["SENTINEL_TESTS_JSON"]}',
    "eval.sh": "SENTINEL_EVAL_SCRIPT",
    "Dockerfile": "SENTINEL_DOCKERFILE",
}
SENTINELS = ("SENTINEL_GOLD", "SENTINEL_HINTS", "SENTINEL_TEST", "SENTINEL_EVAL", "SENTINEL_DOCKER")


class FakeGit:
    def __init__(self, commit: str | None = COMMIT, dirty: bool | None = False):
        self.commit = commit
        self.dirty = dirty
        self.calls: list[Path] = []
        self.repos_by_path: dict[str, tuple[str | None, bool | None]] = {}

    def set_repo(self, path, commit, dirty):
        self.repos_by_path[str(path)] = (commit, dirty)

    def is_dirty(self, repo: Path) -> bool | None:
        return self.repos_by_path.get(str(repo), (None, self.dirty))[1]

    def head_commit(self, repo: Path) -> str | None:
        self.calls.append(repo)
        return self.repos_by_path.get(str(repo), (self.commit, None))[0]


def task_yaml(instance_id: str, repo: str, /, **overrides) -> dict:
    data = {
        "base_commit": "a" * 40,
        "created_at": "2023-02-20 20:13:10",
        "datasets": ["SWE-bench/SWE-bench_Multilingual"],
        "eval_type": "pass_and_fail",  # evaluator-side: must never reach Task
        "image": f"swebench/sweb.eval.x86_64.{instance_id.replace('__', '_1776_')}:latest",
        "instance_id": instance_id,
        "log_parser": "parse_log_jest",  # evaluator-side
        "repo": repo,
        "split": "test",
        "version": "15445",
    }
    data.update(overrides)
    return data


def add_task(
    root: Path,
    instance_id: str,
    repo: str = "babel/babel",
    /,
    *,
    statement: str | bytes | None = "Fix the bug.\n",
    yaml_text: str | None = None,
    hidden: bool = True,
    **overrides,
) -> Path:
    d = root / "tasks" / instance_id
    d.mkdir(parents=True)
    if yaml_text is None:
        yaml_text = yaml.safe_dump(task_yaml(instance_id, repo, **overrides))
    (d / "task.yaml").write_text(yaml_text, encoding="utf-8")
    if isinstance(statement, bytes):
        (d / "problem_statement.md").write_bytes(statement)
    elif statement is not None:
        (d / "problem_statement.md").write_text(statement, encoding="utf-8")
    if hidden:
        for name, content in HIDDEN_FILES.items():
            (d / name).write_text(content)
    return d


def make_task_repo(tmp_path: Path, *, with_tasks: bool = True) -> Path:
    root = tmp_path / "task-repo"
    (root / "tasks").mkdir(parents=True)
    if with_tasks:
        add_task(root, "babel__babel-15445", "babel/babel", statement="Babel bug [x] é\n")
        add_task(root, "axios__axios-5892", "axios/axios")
        add_task(root, "apache__lucene-12212", "apache/lucene")
    return root
