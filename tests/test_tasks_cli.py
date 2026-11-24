import json

import pytest
from typer.testing import CliRunner

from agent_harness_lab.cli import tasks as tasks_cli
from agent_harness_lab.cli.app import app
from agent_harness_lab.doctor.checks import run_checks
from tests.benchmark_fixtures import COMMIT, SENTINELS, FakeGit, add_task, make_task_repo

runner = CliRunner()
WIDE = {"COLUMNS": "200"}


@pytest.fixture(autouse=True)
def fake_git(monkeypatch):
    git = FakeGit()
    monkeypatch.setattr(tasks_cli, "_git_probe", lambda: git)
    monkeypatch.delenv("AHL_SWEBENCH_TASKS_REPO", raising=False)
    return git


@pytest.fixture
def repo(tmp_path):
    return make_task_repo(tmp_path)


def run(*args, env=None):
    return runner.invoke(app, list(args), env={**WIDE, **(env or {})})


# ---- configuration precedence ------------------------------------------------
def test_tasks_repo_option_is_used(repo):
    r = run("tasks", "list", "--tasks-repo", str(repo))
    assert r.exit_code == 0 and "babel__babel-15445" in r.output


def test_cli_option_takes_precedence_over_environment(repo, tmp_path):
    decoy = make_task_repo(tmp_path / "decoy", with_tasks=False)
    add_task(decoy, "decoy__decoy-1", "decoy/decoy")
    r = run("tasks", "list", "--tasks-repo", str(repo), env={"AHL_SWEBENCH_TASKS_REPO": str(decoy)})
    assert "babel__babel-15445" in r.output and "decoy__decoy-1" not in r.output


def test_environment_fallback(repo):
    r = run("tasks", "list", env={"AHL_SWEBENCH_TASKS_REPO": str(repo)})
    assert r.exit_code == 0 and "axios__axios-5892" in r.output


def test_no_configured_repository(repo):
    r = run("tasks", "list")
    assert r.exit_code == 2
    assert "not configured" in r.output and "--tasks-repo" in r.output
    assert "AHL_SWEBENCH_TASKS_REPO" in r.output and "Traceback" not in r.output


def test_invalid_repository_path_is_a_clean_error(tmp_path):
    r = run("tasks", "list", "--tasks-repo", str(tmp_path / "missing"))
    assert r.exit_code == 1 and "does not exist" in r.output and "Traceback" not in r.output


def test_unknown_benchmark(repo):
    r = run("tasks", "list", "--tasks-repo", str(repo), "--benchmark", "nope")
    assert r.exit_code == 1 and "unknown benchmark" in r.output


# ---- tasks list --------------------------------------------------------------
def test_human_list_has_columns_and_no_statement(repo):
    r = run("tasks", "list", "--tasks-repo", str(repo))
    assert r.exit_code == 0
    for text in ("Instance", "Repository", "Base commit", "Docker image", "babel/babel"):
        assert text in r.output
    assert "aaaaaaaaaaaa" in r.output and "a" * 40 not in r.output  # short commit
    assert "Fix the bug" not in r.output and "Babel bug" not in r.output
    assert "3 task(s) shown" in r.output and COMMIT in r.output


def test_repo_filter_is_exact(repo):
    r = run("tasks", "list", "--tasks-repo", str(repo), "--repo", "babel/babel", "--json")
    assert [t["instance_id"] for t in json.loads(r.stdout)["tasks"]] == ["babel__babel-15445"]
    none = run("tasks", "list", "--tasks-repo", str(repo), "--repo", "babel", "--json")
    assert json.loads(none.stdout)["tasks"] == []  # not a substring match


def test_task_filter_is_case_insensitive_substring(repo):
    r = run("tasks", "list", "--tasks-repo", str(repo), "--task", "LUCENE", "--json")
    assert [t["instance_id"] for t in json.loads(r.stdout)["tasks"]] == ["apache__lucene-12212"]


def test_list_json_is_clean_and_complete(repo):
    r = run("tasks", "list", "--tasks-repo", str(repo), "--json")
    data = json.loads(r.stdout)  # entire stdout is JSON
    assert data["benchmark"] == "swebench-multilingual"
    assert data["task_repository"]["commit"] == COMMIT
    assert len(data["tasks"]) == 3 and data["problems"] == []
    t = data["tasks"][0]
    assert {"instance_id", "repo", "base_commit", "docker_image", "task_directory"} <= set(t)
    assert "problem_statement" not in t


def test_malformed_tasks_are_reported_on_stderr_and_kept_out_of_stdout_json(repo):
    add_task(repo, "bad__yaml-1", yaml_text="a: [unclosed")
    r = run("tasks", "list", "--tasks-repo", str(repo), "--json")
    assert r.exit_code == 0
    data = json.loads(r.stdout)
    assert len(data["tasks"]) == 3
    assert data["problems"][0]["instance_id"] == "bad__yaml-1"
    assert "warning: malformed task bad__yaml-1" in r.output  # visible on stderr
    human = run("tasks", "list", "--tasks-repo", str(repo), "--task", "babel")
    assert "malformed task bad__yaml-1" in human.output  # even when filtered out
    assert "1 malformed" in human.output


def test_unreadable_commit_warns(repo, monkeypatch):
    monkeypatch.setattr(tasks_cli, "_git_probe", lambda: FakeGit(None))
    r = run("tasks", "list", "--tasks-repo", str(repo), "--json")
    assert json.loads(r.stdout)["task_repository"]["commit"] is None
    assert "could not read" in r.output


# ---- tasks show --------------------------------------------------------------
def test_show_human(repo):
    r = run("tasks", "show", "babel__babel-15445", "--tasks-repo", str(repo))
    assert r.exit_code == 0
    for text in (
        "swebench-multilingual", "babel__babel-15445", "babel/babel", "a" * 40,
        "swebench/sweb.eval.x86_64.babel_1776_babel-15445:latest", COMMIT,
        "Problem statement", "Babel bug [x] é",
    ):  # fmt: skip
        assert text in r.output  # the statement's [x] is not eaten as Rich markup


def test_show_json(repo):
    r = run("tasks", "show", "babel__babel-15445", "--tasks-repo", str(repo), "--json")
    data = json.loads(r.stdout)
    assert data["instance_id"] == "babel__babel-15445"
    assert data["problem_statement"] == "Babel bug [x] é\n"
    assert data["task_repository_commit"] == COMMIT


def test_show_missing_task(repo):
    r = run("tasks", "show", "nope__nope-1", "--tasks-repo", str(repo))
    assert r.exit_code == 1 and "task not found: nope__nope-1" in r.output


def test_show_malformed_task_is_a_clean_error(repo):
    add_task(repo, "bad__stmt-2", statement=None)
    r = run("tasks", "show", "bad__stmt-2", "--tasks-repo", str(repo))
    assert r.exit_code == 1 and "missing problem_statement.md" in r.output
    assert "Traceback" not in r.output


def test_cli_never_exposes_evaluator_content(repo):
    outputs = [
        run("tasks", "list", "--tasks-repo", str(repo)).output,
        run("tasks", "list", "--tasks-repo", str(repo), "--json").output,
        run("tasks", "show", "babel__babel-15445", "--tasks-repo", str(repo)).output,
        run("tasks", "show", "babel__babel-15445", "--tasks-repo", str(repo), "--json").output,
    ]
    for out in outputs:
        for sentinel in SENTINELS:
            assert sentinel not in out
        assert "pass_and_fail" not in out


def test_list_works_without_docker(repo, monkeypatch):
    import subprocess

    def no_subprocess(*a, **k):
        raise AssertionError("tasks list must not shell out (no Docker, no git in tests)")

    monkeypatch.setattr(subprocess, "run", no_subprocess)
    assert run("tasks", "list", "--tasks-repo", str(repo)).exit_code == 0


# ---- doctor ------------------------------------------------------------------
def test_doctor_without_configured_repo_warns_and_succeeds(monkeypatch, healthy_probe):
    from agent_harness_lab.cli import doctor as doctor_cli

    del healthy_probe.env["AHL_SWEBENCH_TASKS_REPO"]
    monkeypatch.setattr(doctor_cli, "_make_probe", lambda: healthy_probe)
    r = run("doctor")
    assert r.exit_code == 0
    assert "SWE-bench tasks repo" in r.output and "not configured" in r.output


def test_doctor_with_configured_repo_reports_commit(monkeypatch, healthy_probe):
    from agent_harness_lab.cli import doctor as doctor_cli

    monkeypatch.setattr(doctor_cli, "_make_probe", lambda: healthy_probe)
    r = run("doctor")
    assert r.exit_code == 0 and COMMIT in r.output


def test_doctor_report_includes_tasks_repo_check(healthy_probe):
    names = [c.name for c in run_checks(healthy_probe).results]
    assert "SWE-bench tasks repo" in names
