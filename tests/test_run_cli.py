import json
import os
import subprocess

import pytest
from rich.text import Text
from typer.testing import CliRunner

from agent_harness_lab.cli import run as run_cli
from agent_harness_lab.cli.app import app
from tests.plan_fixtures import SECRET

runner = CliRunner()
WIDE = {"COLUMNS": "200"}


@pytest.fixture(autouse=True)
def isolate(monkeypatch, plan_probe, plan_git, tmp_path):
    """Fake host + git; any real subprocess (docker, agent, git) fails the test."""
    monkeypatch.setattr(run_cli, "_probe", lambda: plan_probe)
    monkeypatch.setattr(run_cli, "_git", lambda: plan_git)
    for var in ("AHL_SWEBENCH_TASKS_REPO", "AHL_PROVIDERS_FILE"):
        monkeypatch.delenv(var, raising=False)

    def forbidden(*a, **k):
        raise AssertionError(f"dry-run executed a process: {a!r}")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(os, "system", forbidden)
    work = tmp_path / "cwd"
    work.mkdir()
    monkeypatch.chdir(work)
    return work


def base_args(task_repo, **extra):
    args = [
        "run", "--dry-run", "--task", "babel__babel-15445", "--harness", "pi",
        "--model", "openai/gpt-6-astra", "--reasoning", "medium", "--tasks-repo", str(task_repo),
    ]  # fmt: skip
    for k, v in extra.items():
        args += [f"--{k.replace('_', '-')}", v]
    return args


def invoke(*args):
    return runner.invoke(app, list(args), env=WIDE)


# ---- safety ---------------------------------------------------------------------------
def test_run_without_dry_run_fails_safely(task_repo):
    r = invoke("run", "--task", "babel__babel-15445", "--harness", "pi",
               "--model", "openai/gpt-6-astra", "--tasks-repo", str(task_repo))  # fmt: skip
    assert r.exit_code == 2
    assert "real execution is disabled" in r.output and "AHL_ENABLE_REAL_EXECUTION=1" in r.output


def test_required_options(task_repo):
    r = invoke("run", "--dry-run", "--tasks-repo", str(task_repo))
    assert r.exit_code == 2
    # Typer/Rich may style pieces of the option name (CI); compare the human-visible text
    assert "--task" in Text.from_ansi(r.output).plain


def test_dry_run_executes_nothing_and_writes_nothing(task_repo, isolate):
    before = sorted(p.name for p in task_repo.rglob("*"))
    r = invoke(*base_args(task_repo))  # subprocess/Popen/os.system are patched to explode
    assert r.exit_code == 0, r.output
    assert list(isolate.iterdir()) == []  # no results/raw or other artifact in the cwd
    assert sorted(p.name for p in task_repo.rglob("*")) == before


# ---- human output ---------------------------------------------------------------------
def test_dry_run_human_summary(task_repo):
    r = invoke(*base_args(task_repo))
    assert r.exit_code == 0
    for text in (
        "babel__babel-15445", "swebench-multilingual", "Task repo commit", "Base commit",
        "a" * 40, "babel-15445:latest", "Pi", "1.0.2", "expected pin: 1.0.2",
        "openai/gpt-6-astra", "medium", "3600 s", "restricted-provider", "api.openai.com",
        "OPENAI_API_KEY", "AHL commit", "Dirty state", "Agent invocation",
        "--thinking medium", "<<AHL_PROMPT>>", "nothing was executed",
    ):  # fmt: skip
        assert text in r.output, text
    assert SECRET not in r.output


def test_dry_run_opencode(task_repo):
    r = invoke(*base_args(task_repo, harness="opencode"))
    assert r.exit_code == 0
    assert "OpenCode" in r.output and "openai/gpt-6-astra#medium" in r.output
    assert "--standalone --auto --agent build" in r.output.replace("\n", " ")


def test_missing_api_key_warning_without_leak(task_repo, plan_probe):
    del plan_probe.env["OPENAI_API_KEY"]
    r = invoke(*base_args(task_repo))
    assert r.exit_code == 0
    assert "OPENAI_API_KEY is not set" in r.output and "NOT SET" in r.output


def test_dirty_trees_produce_visible_warnings(task_repo, plan_git):
    from tests.plan_fixtures import AHL_COMMIT, PROJECT_ROOT

    plan_git.set_repo(PROJECT_ROOT, AHL_COMMIT, True)
    r = invoke(*base_args(task_repo))
    assert r.exit_code == 0
    assert "AHL working tree has uncommitted changes" in r.output
    assert "dirty" in r.output


# ---- JSON output ------------------------------------------------------------------------
def test_json_output_is_only_the_plan(task_repo):
    r = invoke(*base_args(task_repo), "--json")
    assert r.exit_code == 0
    plan = json.loads(r.stdout)  # all of stdout is the JSON document
    assert plan["schema_version"] == "ahl-runplan-v0"
    assert plan["task"]["instance_id"] == "babel__babel-15445"
    assert plan["invocation"]["argv"][:3] == [
        "/opt/pi-node_modules/.bin/pi",
        "--provider",
        "openai",
    ]
    assert plan["network"] == {"mode": "restricted-provider", "allowed_hosts": ["api.openai.com"]}
    assert plan["timeout_seconds"] == 3600
    assert plan["required_secret_env"] == ["OPENAI_API_KEY"]
    assert SECRET not in r.output


def test_json_warnings_go_to_stderr_only(task_repo, plan_probe):
    del plan_probe.env["OPENAI_API_KEY"]
    r = invoke(*base_args(task_repo), "--json")
    assert "warning:" in r.output  # visible overall...
    json.loads(r.stdout)  # ...but not part of stdout


def test_cli_options_shape_the_plan(task_repo):
    r = invoke(*base_args(task_repo, timeout="120", network="unrestricted"), "--json")
    plan = json.loads(r.stdout)
    assert plan["timeout_seconds"] == 120
    assert plan["network"] == {"mode": "unrestricted", "allowed_hosts": []}


def test_openai_compatible_via_providers_file(task_repo, providers_file):
    r = invoke(
        "run", "--dry-run", "--json", "--task", "babel__babel-15445", "--harness", "opencode",
        "--model", "example-lab/Org/Model-7B-FP8", "--tasks-repo", str(task_repo),
        "--providers-file", str(providers_file),
    )  # fmt: skip
    plan = json.loads(r.stdout)
    assert plan["provider"]["base_url"] == "https://llm.example.org/v1"
    assert plan["network"]["allowed_hosts"] == ["llm.example.org"]
    assert plan["model"]["model"] == "Org/Model-7B-FP8"
    assert plan["invocation"]["argv"][7] == "example-lab/Org/Model-7B-FP8"
    assert "EXAMPLE_LAB_API_KEY" in plan["required_secret_env"]


def test_providers_file_from_environment(task_repo, providers_file, monkeypatch):
    monkeypatch.setenv("AHL_PROVIDERS_FILE", str(providers_file))
    r = invoke(
        "run", "--dry-run", "--json", "--task", "babel__babel-15445", "--harness", "pi",
        "--model", "example-lab/Org/Model-7B-FP8", "--tasks-repo", str(task_repo),
    )  # fmt: skip
    assert r.exit_code == 0 and json.loads(r.stdout)["provider"]["id"] == "example-lab"


# ---- errors ---------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ({"harness": "aider"}, "unknown harness"),
        ({"model": "nope/m"}, "unknown provider"),
        ({"model": "badref"}, "expected <provider>/<model>"),
        ({"task": "nope__nope-1"}, "task not found"),
        ({"reasoning": "ultra"}, "Pi does not support"),
    ],
)
def test_clean_errors(task_repo, override, fragment):
    args = base_args(task_repo)
    for k, v in override.items():
        args[args.index(f"--{k}") + 1] = v
    r = invoke(*args)
    assert r.exit_code == 1 and fragment in r.output and "Traceback" not in r.output


def test_invalid_timeout_is_rejected(task_repo):
    r = invoke(*base_args(task_repo, timeout="0"))
    assert r.exit_code == 2


def test_no_tasks_repo_configured():
    r = invoke("run", "--dry-run", "--task", "x", "--harness", "pi", "--model", "openai/m")
    assert r.exit_code == 2 and "not configured" in r.output


def test_missing_harness_is_reported(task_repo, plan_probe):
    del plan_probe.executables["pi"]
    r = invoke(*base_args(task_repo))
    assert r.exit_code == 1 and "executable 'pi' not found" in r.output


# ---- timeout and capability through the CLI ---------------------------------------------
def test_dry_run_default_timeout_is_still_3600(task_repo):
    r = invoke(*base_args(task_repo), "--json")
    assert json.loads(r.stdout)["timeout_seconds"] == 3600
    assert "Timeout:                3600 s" in invoke(*base_args(task_repo)).output


def test_no_user_facing_no_timeout_option(task_repo):
    r = invoke(*base_args(task_repo), "--no-timeout")
    assert r.exit_code == 2  # unknown option: None is a plan-level (historical) value only


def test_cli_reports_capability_and_warns_when_unknown(task_repo):
    known = invoke(*base_args(task_repo))
    assert (
        "Reasoning support:      known" in known.output
        and "unknown" not in known.output.split("Constraints")[0]
    )
    args = base_args(task_repo)
    args[args.index("--model") + 1] = "openai/some-future-model"
    unknown = invoke(*args)
    assert unknown.exit_code == 0
    assert "Reasoning support:      unknown" in unknown.output
    assert "reasoning support for openai/some-future-model is unknown" in unknown.output


def test_cli_fails_clearly_when_reasoning_requested_for_unsupported_model(
    task_repo, providers_file
):
    r = invoke(
        "run", "--dry-run", "--task", "babel__babel-15445", "--harness", "opencode",
        "--model", "example-lab/Org/Plain", "--reasoning", "medium",
        "--tasks-repo", str(task_repo), "--providers-file", str(providers_file),
    )  # fmt: skip
    assert r.exit_code == 1 and "not supporting reasoning" in r.output
    assert "Traceback" not in r.output
