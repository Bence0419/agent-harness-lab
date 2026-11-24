from pathlib import Path

from agent_harness_lab.config import resolve_tasks_repo


def test_cli_value_takes_precedence_over_environment():
    r = resolve_tasks_repo(Path("/cli/path"), {"AHL_SWEBENCH_TASKS_REPO": "/env/path"})
    assert (r.path, r.source, r.configured) == (Path("/cli/path"), "cli", True)


def test_environment_fallback():
    r = resolve_tasks_repo(None, {"AHL_SWEBENCH_TASKS_REPO": "/env/path"})
    assert (r.path, r.source) == (Path("/env/path"), "env")


def test_not_configured_when_nothing_given():
    r = resolve_tasks_repo(None, {})
    assert (r.path, r.source, r.configured) == (None, None, False)


def test_blank_environment_value_is_not_configured():
    assert not resolve_tasks_repo(None, {"AHL_SWEBENCH_TASKS_REPO": "   "}).configured


def test_no_directory_searching(tmp_path, monkeypatch):
    # A task repo sitting in the cwd or $HOME must not be discovered implicitly.
    (tmp_path / "swe-bench-multilingual-tasks" / "tasks").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("AHL_SWEBENCH_TASKS_REPO", raising=False)
    assert not resolve_tasks_repo(None).configured


def test_tilde_is_expanded_from_the_environment_not_hardcoded(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    r = resolve_tasks_repo(Path("~/tasks"), {})
    assert r.path == tmp_path / "tasks"
