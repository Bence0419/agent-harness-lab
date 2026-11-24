import pathlib

import pytest

from agent_harness_lab.benchmarks import (
    CommitMismatch,
    InvalidTaskRepository,
    MalformedTask,
    Task,
    TaskNotFound,
    UnknownBenchmark,
    available_benchmarks,
    create_benchmark,
    verify_expected_commit,
)
from agent_harness_lab.benchmarks.swebench_multilingual import SWEBenchMultilingual
from tests.benchmark_fixtures import COMMIT, SENTINELS, FakeGit, add_task, make_task_repo


def adapter(root, git=None):
    return SWEBenchMultilingual(root, git or FakeGit())


# ---- discovery and lookup ----------------------------------------------------
def test_valid_repository_discovery_with_multiple_tasks(tmp_path):
    listing = adapter(make_task_repo(tmp_path)).list_tasks()
    assert [t.instance_id for t in listing.tasks] == [
        "apache__lucene-12212",
        "axios__axios-5892",
        "babel__babel-15445",
    ]
    assert listing.problems == []


def test_task_fields_are_normalized(tmp_path):
    t = adapter(make_task_repo(tmp_path)).get_task("babel__babel-15445")
    assert t.benchmark == "swebench-multilingual"
    assert t.repo == "babel/babel"
    assert t.base_commit == "a" * 40
    assert t.docker_image == "swebench/sweb.eval.x86_64.babel_1776_babel-15445:latest"
    assert t.problem_statement == "Babel bug [x] é\n"
    assert t.task_directory.name == "babel__babel-15445"
    assert t.task_repository_commit == COMMIT
    assert (t.split, t.version, t.created_at) == ("test", "15445", "2023-02-20 20:13:10")


def test_get_missing_task(tmp_path):
    with pytest.raises(TaskNotFound, match="nope__nope-1"):
        adapter(make_task_repo(tmp_path)).get_task("nope__nope-1")


@pytest.mark.parametrize("bad", ["../etc", "a/b", "..", ".hidden", ""])
def test_task_id_cannot_escape_the_tasks_directory(tmp_path, bad):
    with pytest.raises(TaskNotFound, match="invalid task id"):
        adapter(make_task_repo(tmp_path)).get_task(bad)


def test_unconfigured_layout_is_rejected(tmp_path):
    with pytest.raises(InvalidTaskRepository, match="does not exist"):
        adapter(tmp_path / "missing").list_tasks()
    with pytest.raises(InvalidTaskRepository, match="does not look like"):
        adapter(tmp_path).list_tasks()  # exists but has no tasks/


def test_stray_files_and_hidden_dirs_in_tasks_are_ignored(tmp_path):
    root = make_task_repo(tmp_path)
    (root / "tasks" / "README.md").write_text("x")
    (root / "tasks" / ".git").mkdir()
    assert len(adapter(root).list_tasks().tasks) == 3


# ---- malformed tasks ---------------------------------------------------------
@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"yaml_text": "a: [unclosed"}, "invalid YAML"),
        ({"yaml_text": "- just\n- a list\n"}, "not a mapping"),
        ({"yaml_text": "instance_id: t__t-1\nrepo: a/b\nimage: x\n"}, "base_commit"),
        (
            {"yaml_text": "instance_id: t__t-1\nbase_commit: ' '\nrepo: a/b\nimage: x\n"},
            "must not be empty",
        ),
        ({"statement": None}, "missing problem_statement.md"),
        ({"statement": b"\xff\xfe bad bytes"}, "not valid UTF-8"),
        ({"statement": "  \n"}, "empty"),
        ({"instance_id_in_yaml": "other__id-9"}, "does not match directory name"),
    ],
)
def test_malformed_task_has_clear_error(tmp_path, kwargs, fragment):
    root = make_task_repo(tmp_path, with_tasks=False)
    iid = "t__t-1"
    override = {}
    if "instance_id_in_yaml" in kwargs:
        override["instance_id"] = kwargs.pop("instance_id_in_yaml")
    add_task(root, iid, "a/b", **kwargs, **override)
    with pytest.raises(MalformedTask, match=fragment) as exc:
        adapter(root).get_task(iid)
    assert iid in str(exc.value)


def test_missing_task_yaml(tmp_path):
    root = make_task_repo(tmp_path, with_tasks=False)
    d = add_task(root, "t__t-1", "a/b")
    (d / "task.yaml").unlink()
    with pytest.raises(MalformedTask, match="missing task.yaml"):
        adapter(root).get_task("t__t-1")


def test_listing_reports_malformed_tasks_but_still_loads_the_rest(tmp_path):
    root = make_task_repo(tmp_path)
    add_task(root, "bad__yaml-1", yaml_text="a: [unclosed")
    add_task(root, "bad__stmt-2", statement=None)
    listing = adapter(root).list_tasks()
    assert len(listing.tasks) == 3
    reasons = {p.instance_id: p.reason for p in listing.problems}
    assert reasons["bad__stmt-2"] == "missing problem_statement.md"
    assert "invalid YAML" in reasons["bad__yaml-1"]
    assert len(reasons) == 2


def test_yaml_scalars_are_coerced_not_crashing(tmp_path):
    root = make_task_repo(tmp_path, with_tasks=False)
    add_task(root, "t__t-1", "a/b", version=3.1, created_at=None)
    t = adapter(root).get_task("t__t-1")
    assert t.version == "3.1" and t.created_at is None


# ---- repository provenance ---------------------------------------------------
def test_repository_commit_extraction_uses_the_injected_probe(tmp_path):
    git = FakeGit("f" * 40)
    info = adapter(make_task_repo(tmp_path), git).repository_info()
    assert info.commit == "f" * 40 and info.benchmark == "swebench-multilingual"
    assert len(git.calls) == 1


def test_commit_is_not_assumed_to_be_the_research_pin(tmp_path):
    t = adapter(make_task_repo(tmp_path), FakeGit("1" * 40)).get_task("axios__axios-5892")
    assert t.task_repository_commit == "1" * 40 != COMMIT


def test_unreadable_commit_is_none_not_invented(tmp_path):
    info = adapter(make_task_repo(tmp_path), FakeGit(None)).repository_info()
    assert info.commit is None


def test_expected_commit_match_and_no_pin(tmp_path):
    info = adapter(make_task_repo(tmp_path)).repository_info()
    verify_expected_commit(info, COMMIT)
    verify_expected_commit(info, COMMIT.upper())
    verify_expected_commit(info, None)


def test_expected_commit_mismatch_is_explicit(tmp_path):
    info = adapter(make_task_repo(tmp_path), FakeGit("2" * 40)).repository_info()
    with pytest.raises(CommitMismatch) as exc:
        verify_expected_commit(info, COMMIT)
    assert COMMIT in str(exc.value) and "2" * 40 in str(exc.value)


def test_expected_commit_with_unreadable_repo_is_a_mismatch(tmp_path):
    info = adapter(make_task_repo(tmp_path), FakeGit(None)).repository_info()
    with pytest.raises(CommitMismatch, match="unknown"):
        verify_expected_commit(info, COMMIT)


# ---- registry ----------------------------------------------------------------
def test_registry(tmp_path):
    assert available_benchmarks() == ["swebench-multilingual"]
    assert create_benchmark("swebench-multilingual", tmp_path).name == "swebench-multilingual"
    with pytest.raises(UnknownBenchmark, match="swebench-multilingual"):
        create_benchmark("nope", tmp_path)


# ---- benchmark integrity: evaluator/gold data never loaded -------------------
ALLOWED = {"task.yaml", "problem_statement.md"}


def test_only_task_yaml_and_problem_statement_are_ever_opened(tmp_path, monkeypatch):
    root = make_task_repo(tmp_path)
    opened: list[str] = []
    real_open = pathlib.Path.open

    def spy(self, *a, **k):
        opened.append(self.name)
        return real_open(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "open", spy)
    a = adapter(root)
    a.list_tasks()
    a.get_task("babel__babel-15445")
    assert opened and set(opened) <= ALLOWED


def test_task_model_has_no_evaluator_or_gold_fields():
    assert set(Task.model_fields) == {
        "benchmark", "instance_id", "repo", "base_commit", "docker_image",
        "problem_statement", "task_directory", "task_repository_commit",
        "split", "version", "created_at",
    }  # fmt: skip
    for forbidden in ("gold", "patch", "hints", "tests", "eval", "log_parser"):
        assert not any(forbidden in f for f in Task.model_fields), forbidden


def test_evaluator_content_never_appears_in_task_data(tmp_path):
    a = adapter(make_task_repo(tmp_path))
    t = a.get_task("babel__babel-15445")
    blob = t.model_dump_json() + a.list_tasks().model_dump_json()
    for sentinel in SENTINELS:
        assert sentinel not in blob
    assert "pass_and_fail" not in blob and "parse_log_jest" not in blob  # task.yaml eval keys
