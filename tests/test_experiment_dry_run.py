"""``ahl benchmark``: dry-run, the real-execution gate, task-repository pinning and the
experiment-wide pre-flight. No agent, no LLM, no real Docker."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent_harness_lab.cli import common
from agent_harness_lab.cli.app import app
from agent_harness_lab.execution.gate import GATE_ENV
from tests.benchmark_fixtures import COMMIT
from tests.experiment_fixtures import (
    SECRET,
    TASKS,
    experiment_dirs,
    experiment_yaml,
    image_of,
    write_experiment,
)

runner = CliRunner()
WIDE = {"COLUMNS": "240"}
CREATING = {"network_create", "run_proxy", "create_container", "start", "copy_in", "exec", "remove"}


def bench(state, text, *extra, **kw):
    path = write_experiment(state, text)
    args = [
        "benchmark", str(path), "--raw-dir", str(state["raw"]),
        "--experiments-dir", str(state["experiments"]), *extra,
    ]  # fmt: skip
    return runner.invoke(app, args, env=WIDE, **kw)


def assert_nothing_created(state):
    assert not state["raw"].exists() and experiment_dirs(state) == []
    assert not (CREATING & set(state["docker"].ops()))
    assert state["evaluator"].evaluated == []


# ---- dry run ------------------------------------------------------------------------------
def test_dry_run_prints_the_summary_header_and_matrix(exp_env):
    text = experiment_yaml(tasks=TASKS, harnesses=("pi", "opencode"), repetitions=2)
    r = bench(exp_env, text, "--dry-run")
    assert r.exit_code == 0, r.output
    for line in (
        "Experiment: astra-test", "Tasks:      3", "Harnesses:  2", "Models:     1",
        "Repeats:    2", "Runs:       12", "Evaluation: enabled",
    ):  # fmt: skip
        assert line in r.output
    assert "Would be:   astra-test-" in r.output and "READY" in r.output
    assert "nothing was executed" in r.output
    for task in TASKS:
        assert task in r.output


def test_dry_run_is_side_effect_free_and_needs_no_gate(exp_env):
    del exp_env["env"][GATE_ENV]
    r = bench(exp_env, experiment_yaml(tasks=TASKS, harnesses=("pi", "opencode")), "--dry-run")
    assert r.exit_code == 0, r.output
    assert_nothing_created(exp_env)
    assert set(exp_env["docker"].ops()) <= {"image_inspect"}  # read-only inspection only
    assert exp_env["evaluator"].evaluated == []


def test_dry_run_json_describes_every_cell_and_plan(exp_env):
    r = bench(
        exp_env, experiment_yaml(tasks=TASKS, harnesses=("pi", "opencode")), "--dry-run", "--json"
    )
    doc = json.loads(r.stdout)
    assert r.exit_code == 0 and doc["ready"] is True and doc["dry_run"] is True
    assert doc["runs"] == 6 == len(doc["cells"])
    assert (doc["tasks"], doc["harnesses"], doc["models"], doc["repetitions"]) == (3, 2, 1, 1)
    assert doc["evaluation_enabled"] is True
    assert doc["real_execution_gate"] == {"env": GATE_ENV, "enabled": True}
    cell = doc["cells"][0]
    assert cell["cell_id"].startswith("task=apache__lucene-12212;harness=opencode;")
    plan = cell["plan"]
    assert plan["purpose"] == "experiment" and plan["experiment"]["attempt"] == 1
    assert plan["experiment"]["matrix_cell_id"] == cell["cell_id"]
    assert plan["experiment"]["definition_sha256"] == doc["definition_sha256"]
    assert doc["task_repository"]["commit"] == COMMIT and doc["task_repository"]["dirty"] is False
    assert_nothing_created(exp_env)


def test_dry_run_reports_missing_secret_image_and_harness_together(exp_env):
    del exp_env["env"]["OPENAI_API_KEY"]
    exp_env["docker"].images.discard(image_of("axios__axios-5892"))
    exp_env["host"].missing.add("/fake/project/node_modules")
    r = bench(exp_env, experiment_yaml(tasks=TASKS), "--dry-run", "--json")
    doc = json.loads(r.stdout)
    assert r.exit_code == 1 and doc["ready"] is False
    checks = {f["check"] for f in doc["failures"]}
    assert {"secret", "task image"} <= checks
    assert checks & {"provisioning", "provisioning source"}  # the missing host path
    image_failure = next(f for f in doc["failures"] if f["check"] == "task image")
    assert len(image_failure["cells"]) == 1 and "axios__axios-5892" in image_failure["cells"][0]
    assert_nothing_created(exp_env)


def test_dry_run_unknown_harness_provider_and_task_are_pre_flight_failures(exp_env):
    text = experiment_yaml(
        tasks=("babel__babel-15445", "no__such-1"),
        harnesses=("pi", "nonesuch"),
        models=(("ghost", "m", None),),
    )
    r = bench(exp_env, text, "--dry-run", "--json")
    doc = json.loads(r.stdout)
    assert r.exit_code == 1
    failures = {(f["check"]) for f in doc["failures"]}
    assert "task" in failures and "harness" in failures and "plan" in failures
    assert any("no__such-1" in f["detail"] for f in doc["failures"])
    assert any("nonesuch" in f["detail"] for f in doc["failures"])
    assert any("ghost" in f["detail"] for f in doc["failures"])


def test_dry_run_model_validation_through_the_provider_system(exp_env, providers_file, monkeypatch):
    monkeypatch.setenv("AHL_PROVIDERS_FILE", str(providers_file))
    exp_env["env"]["EXAMPLE_LAB_API_KEY"] = "k"
    exp_env["probe"].env["EXAMPLE_LAB_API_KEY"] = "k"
    ok = experiment_yaml(
        models=(("example-lab", "Org/Reasoner", "medium"),), network="restricted-provider"
    )
    r = bench(exp_env, ok, "--dry-run", "--json")
    assert r.exit_code == 0, r.output
    plan = json.loads(r.stdout)["cells"][0]["plan"]
    assert plan["provider"]["id"] == "example-lab" and plan["model"]["model"] == "Org/Reasoner"
    assert plan["network"]["allowed_hosts"] == ["llm.example.org"]
    bad = experiment_yaml(models=(("example-lab", "Org/Plain", "medium"),))  # known: no reasoning
    r = bench(exp_env, bad, "--dry-run", "--json")
    doc = json.loads(r.stdout)
    assert r.exit_code == 1 and any(
        "not supporting reasoning" in f["detail"] for f in doc["failures"]
    )


def test_dry_run_checks_evaluator_prerequisites(exp_env, monkeypatch):
    monkeypatch.delenv("AHL_SWEBENCH_PYTHON")
    r = bench(exp_env, experiment_yaml(), "--dry-run", "--json")
    doc = json.loads(r.stdout)
    assert r.exit_code == 1
    assert any(f["check"] == "evaluation: evaluator python" for f in doc["failures"])
    # disabled evaluation: the evaluator is not needed
    r = bench(exp_env, experiment_yaml(evaluation=False), "--dry-run", "--json")
    doc = json.loads(r.stdout)
    assert (
        r.exit_code == 0 and doc["evaluation_enabled"] is False and doc["evaluation_checks"] == []
    )


def test_dry_run_estimates_the_worst_case_duration(exp_env):
    r = bench(exp_env, experiment_yaml(tasks=TASKS, timeout=1800), "--dry-run")
    assert "3 sequential run(s), at most 1800s" in r.output and "1.5 h" in r.output


def test_dry_run_cannot_be_combined_with_resume(exp_env):
    assert bench(exp_env, experiment_yaml(), "--dry-run", "--resume", "x").exit_code == 2
    assert bench(exp_env, experiment_yaml(), "--retry-failed").exit_code == 2
    assert bench(exp_env, experiment_yaml(), "--json").exit_code == 2  # --json needs --dry-run


# ---- the gate ----------------------------------------------------------------------------
@pytest.mark.parametrize("value", [None, "0", "true", "yes", "", " 1", "2"])
def test_without_the_exact_gate_nothing_starts(exp_env, monkeypatch, value):
    if value is None:
        del exp_env["env"][GATE_ENV]
    else:
        exp_env["env"][GATE_ENV] = value

    def no_docker():
        raise AssertionError("Docker must not even be constructed before the gate")

    monkeypatch.setattr(common, "docker_client", no_docker)
    r = bench(exp_env, experiment_yaml())
    assert r.exit_code == 2 and "real execution is disabled" in r.output and GATE_ENV in r.output
    assert not exp_env["raw"].exists() and experiment_dirs(exp_env) == []


@pytest.mark.parametrize("flag", ["--force", "--yes", "--enable-real-execution", "--no-gate"])
def test_there_is_no_flag_that_bypasses_the_gate(exp_env, flag):
    del exp_env["env"][GATE_ENV]
    assert bench(exp_env, experiment_yaml(), flag).exit_code == 2
    assert exp_env["docker"].calls == []


def test_a_malformed_experiment_is_a_clean_error(exp_env):
    r = bench(exp_env, "schema_version: nope\n", "--dry-run")
    assert r.exit_code == 2 and "unsupported schema_version" in r.output
    assert "Traceback" not in r.output


# ---- task repository pinning ----------------------------------------------------------------
def test_pin_matches(exp_env):
    r = bench(exp_env, experiment_yaml(), "--dry-run")
    assert r.exit_code == 0 and f"pinned {COMMIT}" in r.output


def test_pin_mismatch_fails_before_anything(exp_env):
    r = bench(exp_env, experiment_yaml(pin="b" * 40), "--dry-run", "--json")
    doc = json.loads(r.stdout)
    assert r.exit_code == 1
    pin = next(f for f in doc["failures"] if f["check"] == "task repository pin")
    assert "mismatch" in pin["detail"] and "b" * 40 in pin["detail"]
    r = bench(exp_env, experiment_yaml(pin="b" * 40))  # real launch is refused as well
    assert r.exit_code == 1 and "zero runs were started" in r.output
    assert_nothing_created(exp_env)


def test_unpinned_experiment_is_allowed_but_flagged(exp_env):
    r = bench(exp_env, experiment_yaml(pin=None), "--dry-run")
    assert r.exit_code == 0 and "not pinned" in r.output.replace("\n", " ").lower()


def test_dirty_task_repository_fails_by_default(exp_env):
    exp_env["git"].dirty = True
    r = bench(exp_env, experiment_yaml(), "--dry-run", "--json")
    doc = json.loads(r.stdout)
    assert r.exit_code == 1
    assert any(f["check"] == "task repository state" for f in doc["failures"])
    assert bench(exp_env, experiment_yaml()).exit_code == 1
    assert_nothing_created(exp_env)


def test_unknown_dirty_state_also_fails(exp_env):
    exp_env["git"].dirty = None
    r = bench(exp_env, experiment_yaml(), "--dry-run", "--json")
    assert r.exit_code == 1 and "cannot be determined" in r.output


def test_dirty_override_is_allowed_and_recorded_prominently(exp_env):
    exp_env["git"].dirty = True
    r = bench(exp_env, experiment_yaml(), "--dry-run", "--allow-dirty-task-repo")
    assert r.exit_code == 0 and "DEVELOPMENT OVERRIDE --allow-dirty-task-repo" in r.output
    r = bench(exp_env, experiment_yaml(evaluation=False), "--allow-dirty-task-repo")
    assert r.exit_code == 0, r.output
    (d,) = experiment_dirs(exp_env)
    record = json.loads((d / "experiment.json").read_text())
    assert record["overrides"] == ["--allow-dirty-task-repo"]
    assert record["task_repository"]["dirty"] is True
    assert json.loads((d / "summary.json").read_text())["overrides"] == ["--allow-dirty-task-repo"]
    assert "DEVELOPMENT OVERRIDES" in bench_status(exp_env, d.name).output


def bench_status(state, experiment_id):
    return runner.invoke(
        app, ["experiments", "show", experiment_id, "--experiments-dir", str(state["experiments"])],
        env=WIDE,
    )  # fmt: skip


def test_the_task_repository_is_never_modified(exp_env):
    before = sorted(p.name for p in Path(exp_env["tasks_repo"]).rglob("*"))
    bench(exp_env, experiment_yaml(pin="b" * 40), "--dry-run")
    bench(exp_env, experiment_yaml())
    assert sorted(p.name for p in Path(exp_env["tasks_repo"]).rglob("*")) == before
    assert exp_env["git"].calls  # only read-only probes


# ---- harness version pins ------------------------------------------------------------------
def test_harness_version_mismatch_is_a_pre_flight_failure(exp_env):
    exp_env["probe"].commands[("/fake/pi", "--version")] = (0, "1.0.3")
    r = bench(exp_env, experiment_yaml(), "--dry-run", "--json")
    doc = json.loads(r.stdout)
    assert r.exit_code == 1
    failure = next(f for f in doc["failures"] if f["check"] == "harness version")
    assert "1.0.3" in failure["detail"] and "1.0.2" in failure["detail"]
    r = bench(exp_env, experiment_yaml(), "--dry-run", "--allow-harness-version-mismatch")
    assert r.exit_code == 0, r.output
    assert bench(exp_env, experiment_yaml()).exit_code == 1
    assert_nothing_created(exp_env)


# ---- experiment-wide pre-flight ----------------------------------------------------------------
def test_one_failing_cell_means_zero_runs_and_every_failure_is_reported(exp_env):
    exp_env["docker"].images.discard(image_of("apache__lucene-12212"))  # one failure: last cell
    exp_env["probe"].commands[("/fake/opencode", "--version")] = (0, "opencode v9.9.9")  # another
    r = bench(exp_env, experiment_yaml(tasks=TASKS, harnesses=("pi", "opencode")))
    assert r.exit_code == 1
    assert "zero runs were started" in r.output
    assert "task image" in r.output and "harness version" in r.output  # reported together
    assert_nothing_created(exp_env)
    assert exp_env["docker"].agent_runs == []


def test_preflight_runs_for_all_cells_before_the_first_run(exp_env):
    r = bench(exp_env, experiment_yaml(tasks=TASKS, harnesses=("pi", "opencode"), evaluation=False))
    assert r.exit_code == 0, r.output
    ops = exp_env["docker"].ops()
    first_create = ops.index("create_container")
    inspected = {a[0] for op, a in exp_env["docker"].calls[:first_create] if op == "image_inspect"}
    assert {image_of(t) for t in TASKS} <= inspected  # every task image checked before run 1


def test_missing_gate_secret_in_pre_flight_stops_everything(exp_env):
    del exp_env["env"]["OPENAI_API_KEY"]
    r = bench(exp_env, experiment_yaml(tasks=TASKS))
    assert r.exit_code == 1 and "secret" in r.output
    assert_nothing_created(exp_env)


def test_no_secret_value_in_any_dry_run_output(exp_env):
    for extra in (("--dry-run",), ("--dry-run", "--json")):
        r = bench(exp_env, experiment_yaml(tasks=TASKS, harnesses=("pi", "opencode")), *extra)
        assert SECRET not in r.output
    del exp_env["env"]["OPENAI_API_KEY"]
    assert SECRET not in bench(exp_env, experiment_yaml(), "--dry-run").output


def test_an_unusable_evaluator_stops_a_real_launch_before_any_run(exp_env):
    exp_env["evaluator"].probe_returncode = 1  # swebench cannot be imported
    r = bench(exp_env, experiment_yaml(tasks=TASKS))
    assert r.exit_code == 1 and "zero runs were started" in r.output
    assert "evaluation: evaluator package" in r.output
    assert_nothing_created(exp_env)
    assert exp_env["docker"].agent_runs == []
