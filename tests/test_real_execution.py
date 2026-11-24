"""Phase 6B: the real-execution gate, pre-flight, canary provenance and the executor staying
harness-agnostic. Real Pi/OpenCode plans are driven through the generic executor with a FAKE
Docker; no agent, model API or real Docker daemon is involved."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent_harness_lab.artifacts.legacy import load_run
from agent_harness_lab.cli import common
from agent_harness_lab.cli import run as run_cli
from agent_harness_lab.cli.app import app
from agent_harness_lab.execution.gate import GATE_ENV, real_execution_enabled
from agent_harness_lab.execution.network import SQUID_IMAGE
from tests.fake_docker import FakeDocker
from tests.plan_fixtures import PROJECT_ROOT, SECRET

runner = CliRunner()
WIDE = {"COLUMNS": "220"}
TASK = "babel__babel-15445"
TASK_IMAGE = "swebench/sweb.eval.x86_64.babel_1776_babel-15445:latest"
NODE_BIN = "/opt/hostnode/v22.1.0/bin/node"
NODE_ROOT = "/opt/hostnode/v22.1.0"
PI_REAL = f"{PROJECT_ROOT}/node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js"
OC_REAL = f"{PROJECT_ROOT}/node_modules/@opencode/cli/bin/opencode.exe"


class FakeHost:
    """Fake host filesystem facts: symlinks, node location, and which paths are missing."""

    def __init__(self, missing=()):
        self.missing = set(missing)
        self.links = {"/fake/pi": PI_REAL, "/fake/opencode": OC_REAL}

    def which(self, name):
        return NODE_BIN if name == "node" else None

    def realpath(self, path):
        return self.links.get(path, path)

    def is_file(self, path):
        return path not in self.missing

    def is_dir(self, path):
        return path not in self.missing


@pytest.fixture
def setup(monkeypatch, plan_probe, plan_git, task_repo, tmp_path):
    state = {
        "env": {"OPENAI_API_KEY": SECRET, GATE_ENV: "1"},
        "docker": FakeDocker(images={TASK_IMAGE, SQUID_IMAGE}),
        "host": FakeHost(),
    }
    monkeypatch.setattr(run_cli, "_probe", lambda: plan_probe)
    monkeypatch.setattr(run_cli, "_git", lambda: plan_git)
    monkeypatch.setattr(common, "environ", lambda: state["env"])
    monkeypatch.setattr(common, "docker_client", lambda: state["docker"])
    monkeypatch.setattr(common, "host_environment", lambda: state["host"])
    monkeypatch.setattr(common, "sleep", lambda s: None)
    monkeypatch.setenv("AHL_SWEBENCH_TASKS_REPO", str(task_repo))
    state["raw"] = tmp_path / "raw"
    return state


def invoke(setup, *extra, harness="pi", network="restricted-provider"):
    args = [
        "run", "--task", TASK, "--harness", harness, "--model", "openai/gpt-6-astra",
        "--reasoning", "medium", "--network", network, "--raw-dir", str(setup["raw"]),
        *extra,
    ]  # fmt: skip
    return runner.invoke(app, args, env=WIDE)


CREATING_OPS = {
    "network_create",
    "run_proxy",
    "create_container",
    "start",
    "copy_in",
    "exec",
    "remove",
}


def created_anything(docker: FakeDocker) -> bool:
    return bool(CREATING_OPS & set(docker.ops()))


# ---- the gate ----
def test_gate_absent_refuses_real_execution_before_any_docker_access(setup, monkeypatch):
    del setup["env"][GATE_ENV]

    def no_docker():
        raise AssertionError("Docker client must not even be constructed")

    monkeypatch.setattr(common, "docker_client", no_docker)
    r = invoke(setup)
    assert r.exit_code == 2
    assert "real execution is disabled" in r.output and GATE_ENV in r.output


def test_gate_exact_value_1_allows_execution(setup):
    r = invoke(setup)
    assert r.exit_code == 0, r.output
    assert "State:      completed" in r.output


@pytest.mark.parametrize("value", ["0", "true", "yes", "TRUE", "on", "", " 1", "1 ", "01", "2"])
def test_other_gate_values_do_not_enable_execution(setup, value):
    setup["env"][GATE_ENV] = value
    assert real_execution_enabled({GATE_ENV: value}) is False
    r = invoke(setup)
    assert r.exit_code == 2 and setup["docker"].calls == []


def test_gate_function_accepts_only_exactly_1():
    assert real_execution_enabled({GATE_ENV: "1"}) is True
    assert real_execution_enabled({}) is False


def test_dry_run_never_requires_or_reads_the_gate(setup):
    del setup["env"][GATE_ENV]
    r = invoke(setup, "--dry-run")
    assert r.exit_code == 0 and "nothing was executed" in r.output
    setup["env"][GATE_ENV] = "0"
    assert invoke(setup, "--dry-run").exit_code == 0
    assert setup["docker"].calls == []  # dry-run touches no Docker at all


def test_there_is_no_cli_bypass_for_the_gate(setup):
    del setup["env"][GATE_ENV]
    for flag in ("--force", "--yes", "--enable-real-execution", "--allow-real-harness"):
        assert invoke(setup, flag).exit_code == 2
    assert setup["docker"].calls == []


def test_executor_itself_rechecks_the_gate(tmp_path):
    from agent_harness_lab.execution.executor import RunRequest, execute_run
    from agent_harness_lab.execution.models import ExecutionDisabled
    from tests.mock_harness import mock_plan, mock_provisioner, write_mock_harness

    request = RunRequest(
        plan=mock_plan(harness_name="pi"), prompt="Fix the mock issue.", raw_dir=tmp_path / "raw",
        provisioner=mock_provisioner(write_mock_harness(tmp_path)),
    )  # fmt: skip
    docker = FakeDocker()
    with pytest.raises(ExecutionDisabled):
        execute_run(request, docker=docker, environ={GATE_ENV: "true"})
    assert docker.calls == []
    doc = execute_run(request, docker=docker, environ={GATE_ENV: "1"}, sleep=lambda s: None)
    assert doc.state.value == "completed"


# ---- pre-flight command ----
def test_preflight_is_side_effect_free_and_needs_no_gate(setup):
    del setup["env"][GATE_ENV]
    r = invoke(setup, "--preflight")
    docker = setup["docker"]
    assert set(docker.ops()) <= {"image_inspect"}  # only read-only image inspection
    assert not created_anything(docker) and not setup["raw"].exists()
    assert r.exit_code == 1  # gate not set => not ready, and it says so
    assert "real-execution gate" in r.output and "NOT READY" in r.output


def test_preflight_ready_when_everything_is_in_place(setup):
    r = invoke(setup, "--preflight")
    assert r.exit_code == 0, r.output
    assert "READY" in r.output and "nothing was created or executed" in r.output
    assert not created_anything(setup["docker"])


def test_preflight_json_lists_exact_host_paths_and_no_secrets(setup):
    r = invoke(setup, "--preflight", "--json")
    report = json.loads(r.stdout)
    assert report["ready"] is True
    sources = {p["destination"]: p["source"] for p in report["provisioning"]}
    assert (
        sources["/opt/node24"] == NODE_ROOT
        and sources["/opt/pi-node_modules"] == f"{PROJECT_ROOT}/node_modules"
    )
    assert SECRET not in r.output and report["image"]["reference"] == TASK_IMAGE


def test_preflight_json_reports_not_ready_with_the_failing_checks(setup):
    del setup["env"]["OPENAI_API_KEY"]
    r = invoke(setup, "--preflight", "--json")
    report = json.loads(r.stdout)
    assert r.exit_code == 1 and report["ready"] is False
    assert [c["name"] for c in report["checks"] if c["status"] == "error"] == ["secret"]


def test_preflight_and_dry_run_are_mutually_exclusive(setup):
    assert invoke(setup, "--preflight", "--dry-run").exit_code == 2


def test_preflight_reports_plan_resolution_failures_as_a_failed_check(setup):
    r = runner.invoke(
        app,
        ["run", "--preflight", "--task", "nope__nope-1", "--harness", "pi",
         "--model", "openai/gpt-6-astra", "--raw-dir", str(setup["raw"])],
        env=WIDE,
    )  # fmt: skip
    assert r.exit_code == 1 and "plan resolution" in r.output and "task not found" in r.output


# ---- failures that must create nothing ----
def test_missing_secret_fails_before_any_docker_resource(setup):
    del setup["env"]["OPENAI_API_KEY"]
    r = invoke(setup)
    assert r.exit_code == 1 and "OPENAI_API_KEY is not set" in r.output
    assert not created_anything(setup["docker"]) and not setup["raw"].exists()


def test_missing_task_image_fails_before_container_or_network(setup):
    setup["docker"].images.discard(TASK_IMAGE)
    r = invoke(setup)
    assert r.exit_code == 1 and "not found locally" in r.output
    assert not created_anything(setup["docker"]) and not setup["docker"].networks


def test_missing_proxy_image_fails_preflight_without_unrestricted_fallback(setup):
    setup["docker"].images.discard(SQUID_IMAGE)
    r = invoke(setup)
    assert r.exit_code == 1 and "proxy image" in r.output
    assert not created_anything(setup["docker"])


def test_network_failure_at_execution_time_never_falls_back_to_unrestricted(setup):
    from agent_harness_lab.execution.models import DockerError

    setup["docker"].fail_on["network_create"] = DockerError("cannot create network")
    r = invoke(setup)
    assert r.exit_code == 1 and "stage: network" in r.output
    docker = setup["docker"]
    assert "create_container" not in docker.ops()  # the agent container was never created


def test_canary_requires_restricted_provider_networking(setup):
    r = invoke(setup, network="unrestricted")
    assert r.exit_code == 1 and "canary runs require restricted-provider" in r.output
    assert not created_anything(setup["docker"])


def test_missing_pi_provisioning_source(setup):
    setup["host"].missing.add(f"{NODE_ROOT}/bin/node")
    r = invoke(setup)
    assert r.exit_code == 1 and "does not contain bin/node" in r.output
    assert not created_anything(setup["docker"])


def test_missing_pi_installation_directory(setup):
    setup["host"].missing.add(f"{PROJECT_ROOT}/node_modules")
    r = invoke(setup, "--preflight")
    assert r.exit_code == 1 and "Pi installation directory does not exist" in r.output


def test_missing_opencode_provisioning_source(setup):
    setup["host"].missing.add(OC_REAL)
    r = invoke(setup, harness="opencode")
    assert r.exit_code == 1 and "not a file" in r.output
    assert not created_anything(setup["docker"])


# ---- real harness plans through the GENERIC executor ----
def test_pi_invocation_passes_through_the_generic_executor(setup):
    r = invoke(setup)
    assert r.exit_code == 0, r.output
    ex = [e for e in setup["docker"].exec_log if e["argv"][0] == "/opt/pi-node_modules/.bin/pi"]
    (e,) = ex
    assert e["argv"][1:7] == [
        "--provider",
        "openai",
        "--model",
        "gpt-6-astra",
        "--thinking",
        "medium",
    ]
    assert e["argv"][-1].startswith("Fix the issue described below in this repository.")  # prompt
    assert e["workdir"] == "/testbed" and e["timeout"] == 3600


def test_opencode_invocation_passes_through_the_generic_executor(setup):
    r = invoke(setup, harness="opencode")
    assert r.exit_code == 0, r.output
    (e,) = [x for x in setup["docker"].exec_log if x["argv"][0] == "/usr/local/bin/opencode"]
    assert e["argv"][:9] == [
        "/usr/local/bin/opencode", "run", "--standalone", "--auto", "--agent", "build",
        "--model", "openai/gpt-6-astra#medium", "--format",
    ]  # fmt: skip


def test_real_run_provisioning_and_network_follow_the_plan(setup):
    invoke(setup)
    docker = setup["docker"]
    assert docker.ops().index("network_create") < docker.ops().index("create_container")
    assert [m.destination for m in docker.proxy_mounts] == ["/etc/squid/squid.conf"]
    assert docker.networks == set() and docker.proxies == set()  # all cleaned up


def test_outputs_land_where_the_output_expectation_says(setup):
    invoke(setup)
    (run_dir,) = (setup["raw"] / "swebench" / TASK).iterdir()
    assert {p.name for p in run_dir.iterdir()} >= {
        "run.json", "metadata.json", "prompt.txt", "trace.jsonl", "pi.stderr.log",
        "model.patch", "git-status.txt", "proxy.log",
    }  # fmt: skip
    setup["docker"] = FakeDocker(images={TASK_IMAGE, SQUID_IMAGE})
    invoke(setup, harness="opencode")
    assert any(
        (d / "opencode.stderr.log").exists() for d in (setup["raw"] / "swebench" / TASK).iterdir()
    )


# ---- canary provenance ----
def test_purpose_canary_is_in_plan_run_json_metadata_and_records(setup):
    plan = json.loads(invoke(setup, "--dry-run", "--json").stdout)
    assert plan["purpose"] == "canary"
    invoke(setup)
    (run_dir,) = (setup["raw"] / "swebench" / TASK).iterdir()
    run = json.loads((run_dir / "run.json").read_text())
    assert (
        run["plan"]["purpose"] == "canary"
        and run["plan"]["network"]["mode"] == "restricted-provider"
    )
    assert json.loads((run_dir / "metadata.json").read_text())["purpose"] == "canary"
    rec = load_run(run_dir)
    assert rec.purpose == "canary" and rec.reference is None  # never under a named experiment


def test_run_artifacts_never_contain_the_secret_and_record_provisioning(setup):
    invoke(setup)
    (run_dir,) = (setup["raw"] / "swebench" / TASK).iterdir()
    for f in run_dir.rglob("*"):
        if f.is_file():
            assert SECRET not in f.read_text(), f.name
    run = json.loads((run_dir / "run.json").read_text())
    dests = {p["destination"] for p in run["provisioning"]}
    assert {"/opt/node24", "/opt/pi-node_modules"} <= dests
    assert run["plan"]["required_secret_env"] == ["OPENAI_API_KEY"]  # the NAME only


def test_each_real_run_gets_a_new_directory_never_an_existing_one(setup):
    invoke(setup)
    invoke(setup)
    assert len(list((setup["raw"] / "swebench" / TASK).iterdir())) == 2


def test_compare_excludes_canaries_by_default_but_results_shows_them(setup):
    invoke(setup)
    invoke(setup, harness="opencode")
    raw = str(setup["raw"])
    res = runner.invoke(
        app, ["results", "--raw-dir", raw, "--metrics-source", "none", "--json"], env=WIDE
    )
    assert {r["purpose"] for r in json.loads(res.stdout)} == {"canary"}
    cmp_args = [
        "compare",
        "--raw-dir",
        raw,
        "--metrics-source",
        "none",
        "--harness",
        "Pi",
        "--harness",
        "OpenCode",
    ]
    excluded = runner.invoke(app, cmp_args, env=WIDE)
    assert excluded.exit_code == 1 and "2 canary run(s) excluded" in excluded.output
    included = runner.invoke(app, [*cmp_args, "--include-canaries"], env=WIDE)
    assert included.exit_code == 0 and "Compared 1 task" in included.output


# ---- architecture ----
def test_execution_package_has_no_pi_or_opencode_branching():
    root = Path(__file__).resolve().parents[1] / "src" / "agent_harness_lab" / "execution"
    banned_names = {"pi", "opencode", "Pi", "OpenCode"}
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom | ast.Import):
                names = (
                    [node.module or ""]
                    if isinstance(node, ast.ImportFrom)
                    else [a.name for a in node.names]
                )
                assert not any(n.endswith((".pi", ".opencode")) for n in names), path.name
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert node.value not in banned_names, f"{path.name}: {node.value!r}"
