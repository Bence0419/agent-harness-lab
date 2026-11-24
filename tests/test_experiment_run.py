"""``ahl benchmark`` execution: sequential orchestration, the cell state machine, evaluation,
artifacts, resume, attempts, status and summaries. Fake Docker/evaluator; no agent or LLM."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent_harness_lab.artifacts.ahl_run import load_ahl_run
from agent_harness_lab.artifacts.writer import write_json_atomic
from agent_harness_lab.cli import common
from agent_harness_lab.cli.app import app
from agent_harness_lab.experiments.store import CellState
from tests.experiment_fixtures import (
    SECRET,
    TASKS,
    experiment_dirs,
    experiment_yaml,
    read_json,
    write_experiment,
)

runner = CliRunner()
WIDE = {"COLUMNS": "240"}
SRC = Path(__file__).resolve().parent.parent / "src" / "agent_harness_lab"


def bench(state, text, *extra, name="experiment.yaml"):
    path = write_experiment(state, text, name)
    return runner.invoke(
        app,
        [
            "benchmark",
            str(path),
            "--raw-dir",
            str(state["raw"]),
            "--experiments-dir",
            str(state["experiments"]),
            *extra,
        ],  # fmt: skip
        env=WIDE,
    )


def only_experiment(state) -> Path:
    (d,) = experiment_dirs(state)
    return d


def cells(d: Path) -> dict[str, dict]:
    return {
        c["cell"]["task"] + "/" + c["cell"]["harness"]: c
        for c in read_json(d / "matrix.json")["cells"]
    }


def states(d: Path) -> dict[str, str]:
    return {k: v["state"] for k, v in cells(d).items()}


# ---- artifacts ------------------------------------------------------------------------------
def test_a_run_creates_the_experiment_artifacts(exp_env):
    text = "# keep my comment\n" + experiment_yaml(tasks=TASKS[:2], harnesses=("pi", "opencode"))
    raw_bytes = text.replace("\n", "\r\n").encode()  # CRLF must survive a byte-for-byte copy
    path = exp_env["tmp"] / "crlf.yaml"
    path.write_bytes(raw_bytes)
    r = runner.invoke(
        app,
        ["benchmark", str(path), "--raw-dir", str(exp_env["raw"]),
         "--experiments-dir", str(exp_env["experiments"])],
        env=WIDE,
    )  # fmt: skip
    assert r.exit_code == 0, r.output
    d = only_experiment(exp_env)
    assert (d / "experiment.yaml").read_bytes() == raw_bytes
    record = read_json(d / "experiment.json")
    assert record["schema_version"] == "ahl-experiment-record-v1"
    assert record["definition"]["tasks"] == sorted(TASKS[:2])
    assert record["definition"]["schema_version"] == "ahl-experiment-v1"
    assert record["definition_sha256"] == read_json(d / "summary.json")["definition_sha256"]
    assert d.name == record["experiment_id"]
    assert d.name.endswith(record["definition_sha256"][:8]) and d.name.startswith("astra-test-")
    matrix = read_json(d / "matrix.json")
    assert matrix["schema_version"] == "ahl-matrix-v1" and len(matrix["cells"]) == 4
    assert all(c["state"] == "passed" for c in matrix["cells"])
    assert read_json(d / "summary.json")["status"] == "complete"
    assert not list(d.glob(".*.tmp"))  # no leftover atomic-write temp files
    assert d.parent == exp_env["experiments"] and not str(d).startswith(str(exp_env["raw"]))


def test_experiment_directories_are_never_reused(exp_env):
    from agent_harness_lab.experiments.store import ExperimentRecord, ExperimentStore

    assert bench(exp_env, experiment_yaml(evaluation=False)).exit_code == 0
    d = only_experiment(exp_env)
    store = ExperimentStore(d)
    with pytest.raises(FileExistsError):
        ExperimentStore.create(exp_env["experiments"], store.load_record(), b"x", [])
    assert isinstance(store.load_record(), ExperimentRecord)


def test_artifact_writes_are_atomic(tmp_path, monkeypatch):
    target = tmp_path / "matrix.json"
    write_json_atomic(target, {"v": 1})

    def broken_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken_replace)
    with pytest.raises(OSError):
        write_json_atomic(target, {"v": 2})
    assert json.loads(target.read_text()) == {"v": 1}  # the old content is intact
    assert [p.name for p in tmp_path.iterdir()] == ["matrix.json"]  # temp file cleaned up


def test_matrix_is_persisted_after_every_transition(exp_env, monkeypatch):
    from agent_harness_lab.experiments import store as store_mod

    snapshots: list[dict[str, str]] = []
    original = store_mod.ExperimentStore.save_matrix

    def spy(self, matrix):
        original(self, matrix)
        snapshots.append({c.cell.task: c.state.value for c in matrix.cells})

    monkeypatch.setattr(store_mod.ExperimentStore, "save_matrix", spy)
    bench(exp_env, experiment_yaml(tasks=TASKS[:2], evaluation=True))
    seen = {s["axios__axios-5892"] for s in snapshots}
    assert {"planned", "running", "agent_completed", "passed"} <= seen


# ---- provenance ------------------------------------------------------------------------------
def test_every_run_records_experiment_provenance(exp_env):
    bench(exp_env, experiment_yaml(tasks=TASKS[:2], harnesses=("pi", "opencode"), repetitions=2))
    d = only_experiment(exp_env)
    record = read_json(d / "experiment.json")
    matrix_ids = {c["cell"]["cell_id"] for c in read_json(d / "matrix.json")["cells"]}
    run_dirs = sorted(exp_env["raw"].glob("swebench/*/*"))
    assert len(run_dirs) == 8
    seen = set()
    for rd in run_dirs:
        plan = read_json(rd / "run.json")["plan"]
        ref = plan["experiment"]
        assert plan["purpose"] == "experiment"
        assert ref["experiment_id"] == record["experiment_id"]
        assert ref["definition_sha256"] == record["definition_sha256"]
        assert ref["attempt"] == 1 and ref["repetition"] in (1, 2)
        assert ref["matrix_cell_id"] in matrix_ids
        seen.add(ref["matrix_cell_id"])
        assert read_json(rd / "metadata.json")["purpose"] == "experiment"
        assert load_ahl_run(rd).purpose == "experiment"
    assert seen == matrix_ids  # one run per cell


def test_experiment_runs_are_not_canaries_and_canaries_have_no_experiment(exp_env):
    from agent_harness_lab.runplan.models import RunPlan

    bench(exp_env, experiment_yaml(evaluation=False))
    rd = next(exp_env["raw"].glob("swebench/*/*"))
    data = read_json(rd / "run.json")["plan"]
    assert RunPlan.model_validate(data).purpose == "experiment"
    with pytest.raises(ValueError, match="experiment reference"):
        RunPlan.model_validate({**data, "purpose": "canary"})
    with pytest.raises(ValueError, match="experiment reference"):
        RunPlan.model_validate({**data, "experiment": None})


# ---- sequential execution ------------------------------------------------------------------
def test_cells_run_one_at_a_time_in_matrix_order(exp_env):
    bench(exp_env, experiment_yaml(tasks=TASKS, harnesses=("pi", "opencode"), evaluation=False))
    docker = exp_env["docker"]
    expected = [(t, h) for t in sorted(TASKS) for h in ("opencode", "pi")]
    assert docker.agent_runs == expected
    # a container is created, used and removed before the next one is created
    lifecycle = [op for op in docker.ops() if op in ("create_container", "remove")]
    assert lifecycle == ["create_container", "remove"] * 6


def test_the_orchestration_code_has_no_concurrency():
    forbidden = {"threading", "concurrent", "asyncio", "multiprocessing", "subprocess"}
    for path in (SRC / "experiments").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            assert not forbidden & set(names), f"{path.name} imports {names}"


def test_the_experiments_package_never_branches_on_a_harness_name():
    for path in (SRC / "experiments").glob("*.py"):
        strings = {
            n.value
            for n in ast.walk(ast.parse(path.read_text()))
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        }
        assert not {"pi", "opencode", "Pi", "OpenCode"} & strings, path.name


# ---- evaluation ------------------------------------------------------------------------------
def test_pass_fail_and_no_verdict_are_kept_apart(exp_env):
    exp_env["evaluator"].outcomes = {
        "babel__babel-15445": "unresolved_ids",  # FAIL
        "axios__axios-5892": "resolved_ids",  # PASS
        "apache__lucene-12212": "error_ids",  # evaluator error: no verdict
    }
    r = bench(exp_env, experiment_yaml(tasks=TASKS))
    d = only_experiment(exp_env)
    st = states(d)
    assert st["babel__babel-15445/pi"] == "failed"
    assert st["axios__axios-5892/pi"] == "passed"
    assert st["apache__lucene-12212/pi"] == "no_verdict"
    s = read_json(d / "summary.json")
    assert (
        s["counts"]["passed"] == 1 and s["counts"]["failed"] == 1 and s["counts"]["no_verdict"] == 1
    )
    assert s["correctness"] == {"resolved": 1, "denominator": 2, "success_rate": 0.5}
    assert s["counts"]["infrastructure_failures"] == 1  # evaluator error is not a FAIL
    assert r.exit_code == 0  # every cell reached a terminal state


def test_infra_failure_in_evaluation_is_never_reported_as_fail(exp_env):
    exp_env["evaluator"].outcomes = {t: "error_ids" for t in TASKS}
    bench(exp_env, experiment_yaml(tasks=TASKS))
    d = only_experiment(exp_env)
    s = read_json(d / "summary.json")
    assert s["counts"]["failed"] == 0 and s["counts"]["passed"] == 0
    assert s["correctness"]["denominator"] == 0 and s["correctness"]["success_rate"] is None


def test_evaluation_is_the_existing_service_and_writes_evaluation_json(exp_env):
    bench(exp_env, experiment_yaml())
    rd = next(exp_env["raw"].glob("swebench/*/*"))
    ev = read_json(rd / "evaluation.json")
    assert ev["schema_version"] == "ahl-evaluation-v1" and ev["resolved"] is True
    assert ev["evaluator_image_baseline"]["verification_status"] == "verified"


def test_evaluation_disabled_never_calls_the_evaluator(exp_env):
    r = bench(exp_env, experiment_yaml(tasks=TASKS, evaluation=False))
    d = only_experiment(exp_env)
    assert r.exit_code == 0 and set(states(d).values()) == {"completed"}
    assert exp_env["evaluator"].calls == []  # not even the environment probe
    assert not list(exp_env["raw"].glob("swebench/*/*/evaluation.json"))
    s = read_json(d / "summary.json")
    assert s["counts"]["completed_without_evaluation"] == 3 and s["counts"]["evaluated"] == 0
    assert "evaluat" not in "".join(map(str, s["counts"].get("evaluated_ids", [])))


def test_harness_failure_and_timeout_are_recorded_and_not_evaluated(exp_env):
    exp_env["docker"].script = {
        ("babel__babel-15445", "pi"): "fail",
        ("axios__axios-5892", "pi"): "timeout",
    }
    r = bench(exp_env, experiment_yaml(tasks=TASKS))
    d = only_experiment(exp_env)
    st = states(d)
    assert st["babel__babel-15445/pi"] == "harness_failed"
    assert st["axios__axios-5892/pi"] == "timed_out"
    assert st["apache__lucene-12212/pi"] == "passed"
    assert exp_env["evaluator"].evaluated == ["apache__lucene-12212"]  # rule: only clean runs
    for task in ("babel__babel-15445", "axios__axios-5892"):  # evidence is preserved
        (rd,) = exp_env["raw"].glob(f"swebench/{task}/*")
        assert (rd / "model.patch").is_file() and (rd / "run.json").is_file()
        assert not (rd / "evaluation.json").exists()
    s = read_json(d / "summary.json")
    assert (s["counts"]["harness_failures"], s["counts"]["timeouts"]) == (1, 1)
    assert s["correctness"] == {"resolved": 1, "denominator": 1, "success_rate": 1.0}
    assert r.exit_code == 0


def test_execution_infrastructure_failure_is_its_own_state(exp_env):
    from agent_harness_lab.execution.models import DockerError

    exp_env["docker"].fail_on = {"create_container": DockerError("daemon exploded")}
    bench(exp_env, experiment_yaml(tasks=TASKS[:1], evaluation=False))
    d = only_experiment(exp_env)
    c = next(iter(cells(d).values()))
    assert c["state"] == "infra_failure" and c["attempts"][0]["failed_stage"] == "container"
    assert "daemon exploded" in c["attempts"][0]["error"]
    s = read_json(d / "summary.json")
    assert s["counts"]["infrastructure_failures"] == 1 and s["counts"]["failed"] == 0


def test_evaluation_failing_to_start_leaves_the_cell_awaiting_evaluation(exp_env):
    # healthy at launch (experiment pre-flight), broken by the time the first run finishes
    exp_env["evaluator"].break_after_probes = 1
    r = bench(exp_env, experiment_yaml(tasks=TASKS[:1]))
    d = only_experiment(exp_env)
    c = next(iter(cells(d).values()))
    assert c["state"] == "agent_completed" and "pre-flight failed" in c["attempts"][0]["error"]
    assert read_json(d / "summary.json")["status"] == "incomplete" and r.exit_code == 1
    assert read_json(d / "summary.json")["counts"]["pending"] == 1


# ---- resume ------------------------------------------------------------------------------------
def resume(state, text, experiment_id, *extra):
    return bench(state, text, "--resume", experiment_id, *extra)


def test_resume_skips_completed_cells(exp_env):
    text = experiment_yaml(tasks=TASKS)
    exp_env["docker"].interrupt_on = ("axios__axios-5892", "pi")
    r = bench(exp_env, text)
    assert r.exit_code != 0  # interrupted by "Ctrl-C"
    d = only_experiment(exp_env)
    assert read_json(d / "summary.json")["status"] == "interrupted"
    assert exp_env["docker"].agent_runs == [
        ("apache__lucene-12212", "pi"),
        ("axios__axios-5892", "pi"),
    ]

    exp_env["docker"].interrupt_on = None
    exp_env["docker"].agent_runs.clear()
    r = resume(exp_env, text, d.name)
    assert r.exit_code == 0, r.output
    # lucene (already PASSED) is not rerun; the interrupted cell is an infrastructure failure
    # and is NOT silently rerun either; only the never-started cell runs
    assert exp_env["docker"].agent_runs == [("babel__babel-15445", "pi")]
    st = states(d)
    assert st["apache__lucene-12212/pi"] == "passed"
    assert st["axios__axios-5892/pi"] == "infra_failure"
    assert st["babel__babel-15445/pi"] == "passed"
    assert len(experiment_dirs(exp_env)) == 1  # resumed in place, not a new experiment


def test_resume_evaluates_a_completed_but_unevaluated_run(exp_env):
    text = experiment_yaml(tasks=TASKS[:1])
    exp_env["evaluator"].break_after_probes = 1  # evaluation cannot start the first time
    bench(exp_env, text)
    d = only_experiment(exp_env)
    assert states(d) == {"apache__lucene-12212/pi": "agent_completed"}
    agent_runs_before = list(exp_env["docker"].agent_runs)

    exp_env["evaluator"].break_after_probes, exp_env["evaluator"].probe_returncode = None, 0
    r = resume(exp_env, text, d.name)
    assert r.exit_code == 0, r.output
    assert states(d) == {"apache__lucene-12212/pi": "passed"}
    assert exp_env["docker"].agent_runs == agent_runs_before  # the agent did NOT rerun
    assert len(list(exp_env["raw"].glob("swebench/*/*"))) == 1


def test_interrupted_evaluation_is_resumed_without_rerunning_the_agent(exp_env):
    text = experiment_yaml(tasks=TASKS[:1])
    exp_env["evaluator"].interrupt = True
    bench(exp_env, text)
    d = only_experiment(exp_env)
    assert states(d) == {"apache__lucene-12212/pi": "agent_completed"}
    exp_env["evaluator"].interrupt = False
    assert resume(exp_env, text, d.name).exit_code == 0
    assert states(d) == {"apache__lucene-12212/pi": "passed"}
    assert exp_env["docker"].agent_runs == [("apache__lucene-12212", "pi")]


def test_resume_of_a_finished_experiment_does_nothing(exp_env):
    text = experiment_yaml(tasks=TASKS[:2])
    bench(exp_env, text)
    d = only_experiment(exp_env)
    calls_before, runs = len(exp_env["docker"].calls), list(exp_env["docker"].agent_runs)
    r = resume(exp_env, text, d.name)
    assert r.exit_code == 0 and "already complete" in r.output
    assert exp_env["docker"].agent_runs == runs and len(exp_env["docker"].calls) == calls_before


def test_resume_refuses_a_different_definition(exp_env):
    bench(exp_env, experiment_yaml(tasks=TASKS[:1], evaluation=False))
    d = only_experiment(exp_env)
    r = resume(exp_env, experiment_yaml(tasks=TASKS[:2], evaluation=False), d.name)
    assert r.exit_code == 2 and "does not match" in r.output
    assert len(list(exp_env["raw"].glob("swebench/*/*"))) == 1


def test_resume_unknown_experiment(exp_env):
    r = resume(exp_env, experiment_yaml(), "nope-20260101T000000Z-abcd1234")
    assert r.exit_code == 2 and "experiment not found" in r.output


def test_failed_cells_are_not_rerun_unless_requested(exp_env):
    text = experiment_yaml(tasks=TASKS[:2])
    exp_env["docker"].script = {("axios__axios-5892", "pi"): "fail"}
    bench(exp_env, text)
    d = only_experiment(exp_env)
    assert states(d)["axios__axios-5892/pi"] == "harness_failed"
    exp_env["docker"].agent_runs.clear()
    assert resume(exp_env, text, d.name).exit_code == 0
    assert exp_env["docker"].agent_runs == []  # not rerun automatically


def test_retry_failed_creates_a_new_attempt_and_keeps_the_old_evidence(exp_env):
    text = experiment_yaml(tasks=TASKS[1:2])
    exp_env["docker"].script = {("axios__axios-5892", "pi"): "fail"}
    bench(exp_env, text)
    d = only_experiment(exp_env)
    (first_run,) = exp_env["raw"].glob("swebench/axios__axios-5892/*")
    before = {p.name: p.read_bytes() for p in first_run.iterdir() if p.is_file()}

    exp_env["docker"].script = {}  # the second attempt succeeds
    r = resume(exp_env, text, d.name, "--retry-failed")
    assert r.exit_code == 0, r.output
    cell = next(iter(cells(d).values()))
    assert [a["attempt"] for a in cell["attempts"]] == [1, 2]
    assert cell["attempts"][0]["execution_state"] == "harness_failed"
    assert cell["state"] == "passed"
    runs = sorted(exp_env["raw"].glob("swebench/axios__axios-5892/*"))
    assert len(runs) == 2
    assert {p.name: p.read_bytes() for p in first_run.iterdir() if p.is_file()} == before
    plan = read_json(runs[0] / "run.json")["plan"]["experiment"]
    plans = {read_json(r / "run.json")["plan"]["experiment"]["attempt"] for r in runs}
    assert plans == {1, 2} and plan["matrix_cell_id"]


def test_retry_of_a_no_verdict_cell_reevaluates_instead_of_rerunning_the_agent(exp_env):
    text = experiment_yaml(tasks=TASKS[:1])
    exp_env["evaluator"].outcomes = {"apache__lucene-12212": "error_ids"}
    bench(exp_env, text)
    d = only_experiment(exp_env)
    assert states(d) == {"apache__lucene-12212/pi": "no_verdict"}
    exp_env["evaluator"].outcomes = {}
    r = resume(exp_env, text, d.name, "--retry-failed")
    assert r.exit_code == 0, r.output
    assert states(d) == {"apache__lucene-12212/pi": "passed"}
    assert exp_env["docker"].agent_runs == [("apache__lucene-12212", "pi")]  # agent ran once
    (rd,) = exp_env["raw"].glob("swebench/*/*")
    assert list(rd.glob("evaluation.superseded-*.json"))  # the old evaluation is archived


# ---- resume uses canonical records, not file names --------------------------------------------
def test_resume_does_not_trust_matrix_json_alone(exp_env):
    text = experiment_yaml(tasks=TASKS[:1])
    bench(exp_env, text)
    d = only_experiment(exp_env)
    (rd,) = exp_env["raw"].glob("swebench/*/*")
    (rd / "evaluation.json").unlink()  # matrix.json still says "passed"
    exp_env["docker"].agent_runs.clear()
    assert resume(exp_env, text, d.name).exit_code == 0
    assert exp_env["evaluator"].evaluated.count("apache__lucene-12212") == 2  # re-evaluated
    assert exp_env["docker"].agent_runs == []  # agent run was valid: not rerun


def test_an_invalid_evaluation_json_is_not_completion(exp_env):
    text = experiment_yaml(tasks=TASKS[:1])
    bench(exp_env, text)
    d = only_experiment(exp_env)
    (rd,) = exp_env["raw"].glob("swebench/*/*")
    (rd / "evaluation.json").write_text("{ this is not valid json")
    assert resume(exp_env, text, d.name).exit_code == 0
    assert states(d) == {"apache__lucene-12212/pi": "passed"}
    assert read_json(rd / "evaluation.json")["schema_version"] == "ahl-evaluation-v1"
    assert list(rd.glob("evaluation.superseded-*.json"))  # the broken file was archived, not lost


def test_a_run_record_for_another_cell_is_not_accepted(exp_env):
    text = experiment_yaml(tasks=TASKS[:2])
    bench(exp_env, text)
    d = only_experiment(exp_env)
    (lucene,) = exp_env["raw"].glob("swebench/apache__lucene-12212/*")
    other = read_json(lucene / "run.json")
    other["plan"]["experiment"]["matrix_cell_id"] = "task=someone-else"
    (lucene / "run.json").write_text(json.dumps(other))
    resume(exp_env, text, d.name)
    assert states(d)["apache__lucene-12212/pi"] == "infra_failure"  # provenance mismatch
    assert states(d)["axios__axios-5892/pi"] == "passed"


def test_run_found_by_provenance_when_the_orchestrator_died_before_recording_it(exp_env):
    text = experiment_yaml(tasks=TASKS[:1], evaluation=False)
    bench(exp_env, text)
    d = only_experiment(exp_env)
    matrix = read_json(d / "matrix.json")
    cell = matrix["cells"][0]
    cell["state"], cell["attempts"][0]["run_id"], cell["attempts"][0]["run_path"] = (
        "running",
        None,
        None,
    )  # as if the process was killed right after launching the run
    cell["attempts"][0]["execution_state"] = None
    (d / "matrix.json").write_text(json.dumps(matrix))
    exp_env["docker"].agent_runs.clear()
    assert resume(exp_env, text, d.name).exit_code == 0
    assert states(d) == {"apache__lucene-12212/pi": "completed"}
    assert exp_env["docker"].agent_runs == []  # recognized from run.json, not rerun
    assert next(iter(cells(d).values()))["attempts"][0]["run_id"]


def test_a_cell_stuck_running_with_no_run_record_is_an_infrastructure_failure(exp_env):
    text = experiment_yaml(tasks=TASKS[:1], evaluation=False)
    bench(exp_env, text)
    d = only_experiment(exp_env)
    for rd in exp_env["raw"].glob("swebench/*/*"):
        (rd / "run.json").unlink()
    matrix = read_json(d / "matrix.json")
    matrix["cells"][0]["state"] = "running"
    (d / "matrix.json").write_text(json.dumps(matrix))
    resume(exp_env, text, d.name)
    assert states(d) == {"apache__lucene-12212/pi": "infra_failure"}


# ---- status is offline ------------------------------------------------------------------------
def test_status_works_without_docker_api_key_or_gate(exp_env, monkeypatch):
    bench(exp_env, experiment_yaml(tasks=TASKS[:2], harnesses=("pi", "opencode")))
    d = only_experiment(exp_env)

    def boom(*a, **k):
        raise AssertionError("status must not use external services")

    for name in ("docker_client", "command_runner", "host_environment"):
        monkeypatch.setattr(common, name, boom)
    monkeypatch.setattr(common, "environ", lambda: {})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    r = runner.invoke(
        app,
        ["experiments", "show", d.name, "--experiments-dir", str(exp_env["experiments"])],
        env=WIDE,
    )
    assert r.exit_code == 0, r.output
    assert d.name in r.output and "passed" in r.output and "By harness" in r.output
    assert "2 / 4" not in r.output and "4 / 4" in r.output
    j = runner.invoke(
        app,
        ["experiments", "show", d.name, "--experiments-dir", str(exp_env["experiments"]), "--json"],
        env=WIDE,
    )
    doc = json.loads(j.stdout)
    assert len(doc["matrix"]["cells"]) == 4 and doc["summary"]["status"] == "complete"
    listing = runner.invoke(
        app, ["experiments", "list", "--experiments-dir", str(exp_env["experiments"])]
    )
    assert d.name in listing.stdout


def test_status_of_unknown_or_corrupt_experiment(exp_env):
    r = runner.invoke(
        app, ["experiments", "show", "nope", "--experiments-dir", str(exp_env["experiments"])]
    )
    assert r.exit_code == 1 and "experiment not found" in r.output
    bench(exp_env, experiment_yaml(evaluation=False))
    d = only_experiment(exp_env)
    (d / "matrix.json").write_text("{broken")
    r = runner.invoke(
        app, ["experiments", "show", d.name, "--experiments-dir", str(exp_env["experiments"])]
    )
    assert r.exit_code == 1 and "malformed" in r.output and "Traceback" not in r.output


# ---- summary ---------------------------------------------------------------------------------
def test_summary_aggregates_by_harness_and_model(exp_env):
    exp_env["evaluator"].outcomes = {"babel__babel-15445": "unresolved_ids"}
    exp_env["docker"].script = {("apache__lucene-12212", "opencode"): "timeout"}
    text = experiment_yaml(
        tasks=TASKS,
        harnesses=("pi", "opencode"),
        models=(("openai", "gpt-6-astra", "medium"),),
    )
    bench(exp_env, text)
    s = read_json(only_experiment(exp_env) / "summary.json")
    c = s["counts"]
    assert (c["planned"], c["executed"], c["evaluated"]) == (6, 6, 5)
    assert (c["passed"], c["failed"], c["timeouts"], c["harness_failures"]) == (3, 2, 1, 0)
    assert c["no_verdict"] == 0 and c["pending"] == 0
    assert s["correctness"] == {"resolved": 3, "denominator": 5, "success_rate": 0.6}
    pi, oc = s["by_harness"]["pi"], s["by_harness"]["opencode"]
    assert (pi["resolved"], pi["evaluated"]) == (2, 3) and (oc["resolved"], oc["evaluated"]) == (
        1,
        2,
    )
    assert oc["timeouts"] == 1 and pi["timeouts"] == 0
    (model,) = s["by_model"].values()
    assert (model["resolved"], model["evaluated"]) == (3, 5)
    assert next(iter(s["by_model"])) == "openai/gpt-6-astra (reasoning=medium)"
    assert s["durations"]["agent_duration_seconds"] is not None
    assert s["durations"]["evaluation_duration_seconds"] is not None
    assert "significance" in s["notes"][0]


def test_unavailable_metrics_stay_null_never_zero(exp_env):
    from agent_harness_lab.experiments.store import ExperimentStore
    from agent_harness_lab.experiments.summary import build_summary

    bench(exp_env, experiment_yaml(tasks=TASKS[:1]))
    store = ExperimentStore(only_experiment(exp_env))
    s = build_summary(
        store.load_record(), store.load_matrix(), exp_env["raw"], metrics=None, status="x", now="t"
    )
    totals = s["metrics"]["totals"]
    assert set(totals) >= {"cost_usd", "tool_calls", "model_turns", "cache_read_tokens"}
    trace_derived = ("cost_usd", "tool_calls", "model_turns", "cache_read_tokens", "output_tokens")
    assert all(totals[k] is None for k in trace_derived)  # no metrics source: N/A, never 0
    assert s["counts"]["passed"] == 1  # correctness does not depend on metrics


def test_summary_is_incrementally_recoverable(exp_env):
    exp_env["docker"].interrupt_on = ("axios__axios-5892", "pi")
    bench(exp_env, experiment_yaml(tasks=TASKS))
    s = read_json(only_experiment(exp_env) / "summary.json")
    assert s["status"] == "interrupted"
    assert s["counts"]["passed"] == 1 and s["counts"]["planned"] == 3  # already-finished work kept


def test_repetitions_count_as_separate_cells(exp_env):
    bench(exp_env, experiment_yaml(tasks=TASKS[:1], repetitions=3, evaluation=False))
    s = read_json(only_experiment(exp_env) / "summary.json")
    assert s["counts"]["planned"] == 3 and s["counts"]["executed"] == 3
    assert len(list(exp_env["raw"].glob("swebench/*/*"))) == 3


# ---- security --------------------------------------------------------------------------------
def test_no_secret_value_anywhere_in_artifacts_or_output(exp_env):
    exp_env["docker"].script = {("babel__babel-15445", "pi"): "fail"}
    r = bench(exp_env, experiment_yaml(tasks=TASKS, harnesses=("pi", "opencode")))
    assert SECRET not in r.output
    d = only_experiment(exp_env)
    files = [p for p in d.rglob("*") if p.is_file()] + [
        p for p in exp_env["raw"].rglob("*") if p.is_file()
    ]
    assert files
    for path in files:
        assert SECRET.encode() not in path.read_bytes(), path
    show = runner.invoke(
        app,
        ["experiments", "show", d.name, "--experiments-dir", str(exp_env["experiments"]), "--json"],
    )
    assert SECRET not in show.output
    plan = read_json(next(exp_env["raw"].glob("swebench/*/*")) / "run.json")["plan"]
    assert plan["required_secret_env"] == ["OPENAI_API_KEY"]  # a NAME, never a value


def test_experiment_record_contains_only_env_names(exp_env):
    bench(exp_env, experiment_yaml(evaluation=False))
    record = (only_experiment(exp_env) / "experiment.json").read_text()
    assert "OPENAI_API_KEY" not in record and "sk-" not in record


def test_the_gate_is_still_rechecked_inside_the_executor(exp_env):
    from agent_harness_lab.cli import benchmark as bench_cli

    env_for_executor = {"OPENAI_API_KEY": SECRET}  # gate absent at execution time
    exp_env["env"]["AHL_ENABLE_REAL_EXECUTION"] = "1"
    original = bench_cli.RunnerDeps

    def deps_without_gate(**kw):
        kw["environ"] = env_for_executor
        return original(**kw)

    import pytest as _pytest

    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(bench_cli, "RunnerDeps", deps_without_gate)
        r = bench(exp_env, experiment_yaml(evaluation=False))
    assert r.exit_code != 0
    assert exp_env["docker"].agent_runs == []  # the executor refused: nothing ran
    d = only_experiment(exp_env)
    assert states(d) == {"babel__babel-15445/pi": CellState.PLANNED.value}
