import ast
import json
import os
import pathlib

import pytest

from agent_harness_lab.artifacts.hashing import sha256_file
from agent_harness_lab.artifacts.legacy import load_evaluation, load_run
from agent_harness_lab.artifacts.models import EvaluationState
from agent_harness_lab.benchmarks.evaluator_data import (
    BUILD_FILES,
    HASHED_FILES,
    locate_evaluator_files,
)
from agent_harness_lab.evaluation.base import EvaluationError, RunNotFound, UnsupportedBenchmark
from agent_harness_lab.evaluation.environment import (
    EvaluatorEnvironmentError,
    detect_environment,
    scrubbed_env,
)
from agent_harness_lab.evaluation.models import Classification, resolved_for
from agent_harness_lab.evaluation.registry import available_evaluators, get_evaluator
from agent_harness_lab.evaluation.service import (
    load_subject,
    preflight,
    resolve_run_dir,
    run_evaluation,
)
from agent_harness_lab.evaluation.swebench_multilingual import classify_report
from tests.benchmark_fixtures import COMMIT, FakeGit
from tests.eval_fixtures import (
    KEY,
    TASK,
    TASK_IMAGE,
    FakeEvaluatorRunner,
    make_ahl_run,
    make_ctx,
    make_python,
    report_for,
)
from tests.factories import make_run
from tests.fake_docker import PATCH_TEXT, FakeDocker

ADAPTER = get_evaluator("swebench-multilingual")


@pytest.fixture
def env(tmp_path, task_repo):
    raw = tmp_path / "raw"
    return {
        "raw": raw,
        "python": make_python(tmp_path),
        "repo": task_repo,
        "run_dir": make_ahl_run(tmp_path, raw),
        "tmp": tmp_path,
    }


def ctx_for(env, runner=None, **kw):
    return make_ctx(
        env["run_dir"], env["repo"], env["python"], runner or FakeEvaluatorRunner(), **kw
    )


def names(pre, status):
    return [c.name for c in pre.checks if c.status.value == status]


# ---- registry and run resolution ----
def test_evaluator_registry():
    assert available_evaluators() == ["swebench-multilingual"]
    assert get_evaluator("swebench-multilingual").name == "swebench-multilingual"


def test_unsupported_benchmark():
    with pytest.raises(UnsupportedBenchmark, match="no evaluator for benchmark 'other'"):
        get_evaluator("other")


def test_run_resolved_by_id_and_by_path(env):
    run_id = env["run_dir"].name
    assert resolve_run_dir(env["raw"], run_id) == env["run_dir"]
    assert resolve_run_dir(env["raw"], str(env["run_dir"])) == env["run_dir"]


def test_missing_run(env):
    with pytest.raises(RunNotFound, match="run not found: nope"):
        resolve_run_dir(env["raw"], "nope")
    with pytest.raises(RunNotFound):
        resolve_run_dir(env["tmp"] / "no-raw-dir", "nope")


def test_subject_from_run_json_has_no_harness_information(env):
    s = load_subject(env["run_dir"])
    assert (s.benchmark, s.instance_id, s.docker_image) == (
        "swebench-multilingual",
        TASK,
        TASK_IMAGE,
    )
    assert s.schema_name == "ahl-run-v1" and s.task_repository_commit == COMMIT
    assert set(type(s).model_fields) == {
        "run_id", "run_dir", "schema_name", "benchmark", "instance_id", "docker_image",
        "task_repository_commit", "agent_image_id", "patch_path",
    }  # fmt: skip


def test_missing_patch_is_rejected(env):
    (env["run_dir"] / "model.patch").unlink()
    with pytest.raises(EvaluationError, match="no model.patch"):
        load_subject(env["run_dir"])


@pytest.mark.parametrize("content", ["{broken", '{"schema_version": "ahl-run-v9"}'])
def test_unsupported_run_schema_is_rejected(env, content):
    (env["run_dir"] / "run.json").write_text(content)
    with pytest.raises(EvaluationError, match="unsupported or malformed run.json"):
        load_subject(env["run_dir"])


def test_historical_run_is_evaluable_when_it_has_enough_information(tmp_path):
    run = make_run(tmp_path / "raw", "hist-1", "pi-swebench-reference", instanceId=TASK)
    (run / "model.patch").write_text("diff --git a/x b/x\n")
    s = load_subject(run)
    assert s.schema_name == "pi-swebench-reference" and s.instance_id == TASK
    assert s.benchmark == "swebench-multilingual"


def test_unrecognised_legacy_metadata_is_rejected(tmp_path):
    d = tmp_path / "raw" / "swebench" / "t" / "r"
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"runId": "r", "instanceId": "t"}))
    (d / "model.patch").write_text("")
    with pytest.raises(EvaluationError, match="unsupported run schema"):
        load_subject(d)


# ---- evaluator environment ----
def test_environment_detection_records_python_and_package_version(env):
    prov = detect_environment(env["python"], FakeEvaluatorRunner(version="5.0.2"), {})
    assert prov.evaluator_version == "5.0.2" and prov.python_version == "3.12.3"
    assert prov.python_executable == str(env["python"]) and prov.cli.endswith("swebench")


def test_environment_detection_reports_an_unimportable_evaluator(env):
    with pytest.raises(EvaluatorEnvironmentError, match="cannot be imported.*swebench"):
        detect_environment(env["python"], FakeEvaluatorRunner(probe_returncode=1), {})


def test_scrubbed_env_drops_credentials_only():
    out = scrubbed_env({"OPENAI_API_KEY": "x", "BME_LAB_API_KEY": "y", "GH_TOKEN": "z",
                        "DOCKER_HOST": "unix:///v", "PATH": "/bin", "HOME": "/h"})  # fmt: skip
    assert out == {"DOCKER_HOST": "unix:///v", "PATH": "/bin", "HOME": "/h"}


def test_a_newer_evaluator_version_is_recorded_not_rejected(env):
    pre = preflight(ADAPTER, ctx_for(env, FakeEvaluatorRunner(version="9.9.9")))
    assert pre.ready and pre.environment.evaluator_version == "9.9.9"


# ---- pre-flight ----
def test_preflight_ready(env):
    pre = preflight(ADAPTER, ctx_for(env))
    assert pre.ready, [(c.name, c.detail) for c in pre.checks if c.status.value == "error"]
    assert {"run", "model.patch", "task repository", "task", "evaluator task files",
            "evaluator python", "evaluator CLI", "evaluator package", "task image",
            "existing evaluation"} <= {c.name for c in pre.checks}  # fmt: skip


def test_preflight_creates_nothing_and_runs_no_tests(env):
    runner, before = FakeEvaluatorRunner(), sorted(p.name for p in env["run_dir"].rglob("*"))
    preflight(ADAPTER, ctx_for(env, runner))
    assert sorted(p.name for p in env["run_dir"].rglob("*")) == before
    assert all(
        c["argv"][1:3] == ["-c", c["argv"][2]] for c in runner.calls
    )  # only the import probe


def test_preflight_missing_evaluator_python(env):
    ctx = ctx_for(env)
    ctx.python = env["tmp"] / "nowhere" / "python"
    assert "evaluator python" in names(preflight(ADAPTER, ctx), "error")
    ctx.python = None
    pre = preflight(ADAPTER, ctx)
    assert any("not configured" in c.detail for c in pre.checks if c.name == "evaluator python")


def test_preflight_missing_evaluator_cli(env):
    (env["python"].parent / "swebench").unlink()
    assert "evaluator CLI" in names(preflight(ADAPTER, ctx_for(env)), "error")


def test_preflight_evaluator_import_failure(env):
    pre = preflight(ADAPTER, ctx_for(env, FakeEvaluatorRunner(probe_returncode=1)))
    assert "evaluator package" in names(pre, "error") and not pre.ready


def test_preflight_missing_evaluator_task_files(env):
    (env["repo"] / "tasks" / TASK / "tests.json").unlink()
    pre = preflight(ADAPTER, ctx_for(env))
    detail = next(c.detail for c in pre.checks if c.name == "evaluator task files")
    assert "tests.json" in detail and not pre.ready


def test_preflight_missing_image(env):
    pre = preflight(ADAPTER, ctx_for(env, docker=FakeDocker(images=set())))
    assert "task image" in names(pre, "error")


def test_preflight_unknown_task_repository_commit(env):
    pre = preflight(ADAPTER, ctx_for(env, git=FakeGit(None)))
    assert "task repository" in names(pre, "error")


def test_preflight_warns_when_task_repo_differs_from_the_run(env):
    pre = preflight(ADAPTER, ctx_for(env, git=FakeGit("9" * 40)))
    assert pre.ready and "task repository version" in names(pre, "warning")


def test_preflight_unreadable_patch_encoding(env):
    (env["run_dir"] / "model.patch").write_bytes(b"\xff\xfe\x00bad")
    assert "model.patch" in names(preflight(ADAPTER, ctx_for(env)), "error")


def test_preflight_refuses_existing_evaluation_unless_forced(env):
    (env["run_dir"] / "evaluation.json").write_text("{}")
    assert "existing evaluation" in names(preflight(ADAPTER, ctx_for(env)), "error")
    forced = preflight(ADAPTER, ctx_for(env, force=True))
    assert forced.ready and "existing evaluation" in names(forced, "warning")


# ---- official report classification ----
@pytest.mark.parametrize(
    ("bucket", "state"),
    [
        ("resolved_ids", EvaluationState.RESOLVED),
        ("unresolved_ids", EvaluationState.UNRESOLVED),
        ("empty_patch_ids", EvaluationState.EMPTY_PATCH),
        ("incomplete_ids", EvaluationState.INCOMPLETE),
        # an `error` with no recognised cause stays an evaluator error (no verdict)
        ("error_ids", EvaluationState.ERROR),
    ],
)
def test_primary_buckets_map_to_states(bucket, state):
    got, official, _, _ = classify_report(
        report_for("t", bucket), "t", exit_code=0, timed_out=False
    )
    assert got is state and official["buckets"] == [state.value]


@pytest.mark.parametrize("tag", ["infra_failure_ids", "ambiguous_failure_ids"])
def test_a_tag_without_a_primary_bucket_is_inconsistent_not_a_verdict(tag):
    r = report_for("t", None, tags=(tag,))
    state, _, _, error = classify_report(r, "t", exit_code=0, timed_out=False)
    assert state is EvaluationState.UNKNOWN and "absent from every primary" in error


def test_resolved_requires_clean_completion():
    r = report_for("t", "resolved_ids", completed=False)
    state, _, _, error = classify_report(r, "t", exit_code=0, timed_out=False)
    assert state is EvaluationState.AMBIGUOUS_FAILURE and "not trusting" in error
    state, *_ = classify_report(report_for("t", "resolved_ids"), "t", exit_code=1, timed_out=False)
    assert state is EvaluationState.AMBIGUOUS_FAILURE  # success is never inferred


def test_instance_in_several_lists_is_ambiguous():
    r = report_for("t", "resolved_ids")
    r["unresolved_ids"] = ["t"]
    state, _, _, error = classify_report(r, "t", exit_code=0, timed_out=False)
    assert state is EvaluationState.AMBIGUOUS_FAILURE and "several primary" in error


def test_instance_in_no_list_has_no_verdict():
    state, _, _, error = classify_report(report_for("t", None), "t", exit_code=0, timed_out=False)
    assert state is EvaluationState.UNKNOWN and "absent from every primary" in error


def test_missing_report_and_timeout_are_infrastructure_failures():
    state, _, _, error = classify_report(None, "t", exit_code=2, timed_out=False)
    assert state is EvaluationState.INFRA_FAILURE and "no readable report" in error
    state, _, _, error = classify_report(
        report_for("t", "resolved_ids"), "t", exit_code=0, timed_out=True
    )
    assert state is EvaluationState.INFRA_FAILURE and "timed out" in error  # even with a report


def test_unresolved_with_failing_exit_is_kept_with_a_warning():
    state, _, warnings, _ = classify_report(
        report_for("t", "unresolved_ids"), "t", exit_code=3, timed_out=False
    )
    assert state is EvaluationState.UNRESOLVED and "exited with code 3" in warnings[0]


def test_failure_reason_is_carried_into_the_official_result():
    r = report_for("t", "error_ids", extra={"failure_reasons": {"t": "image build failed"}})
    _, official, _, _ = classify_report(r, "t", exit_code=0, timed_out=False)
    assert official["failure_reason"] == "image build failed"


@pytest.mark.parametrize(
    ("state", "resolved", "cls"),
    [
        (EvaluationState.RESOLVED, True, Classification.BENCHMARK_PASS),
        (EvaluationState.UNRESOLVED, False, Classification.BENCHMARK_FAIL),
        (EvaluationState.EMPTY_PATCH, False, Classification.BENCHMARK_FAIL),
        (EvaluationState.INFRA_FAILURE, None, Classification.EVALUATION_FAILURE),
        (EvaluationState.ERROR, None, Classification.EVALUATION_FAILURE),
        (EvaluationState.INCOMPLETE, None, Classification.EVALUATION_FAILURE),
        (EvaluationState.AMBIGUOUS_FAILURE, None, Classification.AMBIGUOUS),
        (EvaluationState.UNKNOWN, None, Classification.AMBIGUOUS),
    ],
)
def test_infrastructure_failure_is_never_a_benchmark_fail(state, resolved, cls):
    from agent_harness_lab.evaluation.models import classify

    assert resolved_for(state) is resolved and classify(state) is cls


# ---- end-to-end evaluation (fake evaluator process) ----
def evaluate(env, runner=None, **kw):
    ctx = ctx_for(env, runner, **kw)
    return ctx, *run_evaluation(ADAPTER, ctx)


def test_pass_end_to_end(env):
    runner = FakeEvaluatorRunner(bucket="resolved_ids")
    ctx, pre, doc = evaluate(env, runner)
    assert doc.state is EvaluationState.RESOLVED and doc.resolved is True
    assert doc.classification is Classification.BENCHMARK_PASS and doc.error is None
    on_disk = json.loads((env["run_dir"] / "evaluation.json").read_text())
    assert on_disk["schema_version"] == "ahl-evaluation-v1" and on_disk["resolved"] is True
    assert (
        on_disk["run_id"] == env["run_dir"].name and on_disk["benchmark"] == "swebench-multilingual"
    )
    assert on_disk["evaluator_version"] == "5.0.2" and on_disk["task_repository_commit"] == COMMIT
    assert on_disk["image"]["reference"] == TASK_IMAGE


def test_fail_end_to_end(env):
    _, _, doc = evaluate(env, FakeEvaluatorRunner(bucket="unresolved_ids"))
    assert doc.resolved is False and doc.classification is Classification.BENCHMARK_FAIL


def test_empty_patch_end_to_end(env):
    _, _, doc = evaluate(env, FakeEvaluatorRunner(bucket="empty_patch_ids"))
    assert doc.state is EvaluationState.EMPTY_PATCH and doc.resolved is False


def test_infrastructure_failure_end_to_end_has_no_verdict(env):  # no report at all
    _, _, doc = evaluate(env, FakeEvaluatorRunner(report=None, returncode=1))
    assert doc.state is EvaluationState.INFRA_FAILURE and doc.resolved is None
    assert (
        doc.classification is Classification.EVALUATION_FAILURE
        and "no readable report" in doc.error
    )
    assert (env["run_dir"] / "evaluation.json").exists()  # evidence is still persisted


def test_evaluator_cannot_start_is_an_infrastructure_failure(env):
    _, _, doc = evaluate(env, FakeEvaluatorRunner(start_error="exec format error"))
    assert doc.state is EvaluationState.INFRA_FAILURE and "could not start" in doc.error


def test_ambiguous_output_end_to_end(env):
    _, _, doc = evaluate(
        env, FakeEvaluatorRunner(bucket="resolved_ids", extra_buckets=("unresolved_ids",))
    )
    assert doc.state is EvaluationState.AMBIGUOUS_FAILURE and doc.resolved is None


def test_process_timeout_is_an_infrastructure_failure(env):
    _, _, doc = evaluate(env, FakeEvaluatorRunner(timed_out=True, returncode=-15))
    assert doc.state is EvaluationState.INFRA_FAILURE and doc.command.timed_out


def test_official_command_construction(env):
    runner = FakeEvaluatorRunner()
    ctx, _, doc = evaluate(env, runner)
    (call,) = [c for c in runner.calls if "eval" in c["argv"]]
    a, work = call["argv"], env["run_dir"] / "evaluator"
    assert a[:5] == [
        str(env["python"]),
        str(env["python"].parent / "swebench"),
        "eval",
        "multilingual",
        "-p",
    ]
    assert (
        a[a.index("-i") + 1] == TASK
        and a[a.index("-j") + 1] == "1"
        and a[a.index("-t") + 1] == "1800"
    )
    assert a[a.index("--task-repo") + 1] == str(env["repo"]) and a[
        a.index("--report-dir") + 1
    ] == str(work)
    assert a[a.index("--run-id") + 1].startswith(f"ahl-eval-{env['run_dir'].name}-")
    assert call["cwd"] == work  # evaluator output stays inside the run's own directory
    assert doc.command.argv == a and doc.command.eval_run_id == a[a.index("--run-id") + 1]


def test_predictions_file_carries_the_exact_persisted_patch(env):
    runner = FakeEvaluatorRunner()
    evaluate(env, runner)
    pred = json.loads((env["run_dir"] / "evaluator" / "predictions.jsonl").read_text())
    assert pred["model_patch"] == (env["run_dir"] / "model.patch").read_text() == PATCH_TEXT
    assert pred["instance_id"] == TASK


def test_patch_and_agent_artifacts_are_never_modified(env):
    protected = ["model.patch", "run.json", "prompt.txt", "trace.jsonl", "metadata.json"]
    before = {n: (env["run_dir"] / n).read_bytes() for n in protected}
    evaluate(env)
    assert {n: (env["run_dir"] / n).read_bytes() for n in protected} == before


def test_artifacts_are_hashed_and_named(env):
    _, _, doc = evaluate(env)
    assert {"evaluator.stdout.log", "evaluator.stderr.log"} <= set(doc.artifacts)
    assert any(
        k.startswith("evaluator/") and k.endswith("predictions.jsonl") for k in doc.artifacts
    )
    assert any(k.endswith(".json") and "ahl-" in k for k in doc.artifacts)  # the official report
    for rel, digest in doc.artifacts.items():
        assert sha256_file(env["run_dir"] / rel) == digest
    assert doc.patch_sha256 == sha256_file(env["run_dir"] / "model.patch")
    assert set(doc.evaluator_inputs_sha256) == set(HASHED_FILES)


def test_evaluation_duration_is_separate_from_agent_duration(env):
    run_before = json.loads((env["run_dir"] / "run.json").read_text())
    ctx = ctx_for(env, step=7)  # the evaluation clock ticks 7 s; the agent run used 1 s
    _, doc = run_evaluation(ADAPTER, ctx)
    assert doc.duration_seconds == 7.0 and run_before["duration_seconds"] == 1.0
    assert json.loads((env["run_dir"] / "run.json").read_text())["duration_seconds"] == 1.0
    assert load_run(env["run_dir"]).duration_seconds == 1  # results still report agent time


def test_no_secret_dependency_and_no_secret_reaches_the_evaluator(env):
    runner = FakeEvaluatorRunner()
    environ = {"OPENAI_API_KEY": KEY, "PATH": "/usr/bin"}  # no execution gate set either
    _, _, doc = evaluate(env, runner, environ=environ)
    assert doc.resolved is True
    eval_call = next(c for c in runner.calls if "eval" in c["argv"])
    assert "OPENAI_API_KEY" not in eval_call["env"] and eval_call["env"]["PATH"] == "/usr/bin"
    for f in env["run_dir"].rglob("*"):
        if f.is_file():
            assert KEY not in f.read_text(errors="ignore")


# ---- overwrite policy, atomicity ----
def test_existing_evaluation_is_refused_and_untouched(env):
    (env["run_dir"] / "evaluation.json").write_text('{"keep": "me"}')
    _, pre, doc = evaluate(env)
    assert doc is None and not pre.ready
    assert (env["run_dir"] / "evaluation.json").read_text() == '{"keep": "me"}'
    assert not (env["run_dir"] / "evaluator").exists()


def test_force_archives_the_previous_evaluation_instead_of_overwriting(env):
    evaluate(env, FakeEvaluatorRunner(bucket="unresolved_ids"))
    first = (env["run_dir"] / "evaluation.json").read_text()
    _, _, doc = evaluate(env, FakeEvaluatorRunner(bucket="resolved_ids"), force=True)
    archived = sorted(env["run_dir"].glob("evaluation.superseded-*.json"))
    assert len(archived) == 1 and archived[0].read_text() == first  # old evidence preserved
    assert doc.replaces == archived[0].name and doc.resolved is True
    assert sorted(env["run_dir"].glob("evaluator.superseded-*"))  # old artifacts kept too
    assert (
        json.loads((env["run_dir"] / "evaluation.json").read_text())["replaces"] == archived[0].name
    )


def test_evaluation_json_is_written_atomically(env, monkeypatch):
    monkeypatch.setattr(os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        evaluate(env)
    assert not (env["run_dir"] / "evaluation.json").exists()  # never a half-written file
    assert not list(env["run_dir"].glob(".evaluation.json.*"))


def test_output_dir_keeps_the_source_run_untouched(env):
    out = env["tmp"] / "scratch"
    before = sorted(p.name for p in env["run_dir"].rglob("*"))
    _, _, doc = evaluate(env, output_dir=out)
    assert (out / "evaluation.json").exists()
    assert sorted(p.name for p in env["run_dir"].rglob("*")) == before  # nothing written there


# ---- result integration ----
@pytest.mark.parametrize(
    ("bucket", "tags", "expected"),
    [
        ("resolved_ids", (), True),
        ("unresolved_ids", (), False),
        ("unresolved_ids", ("infra_failure_ids",), None),  # official infra tag: no verdict
    ],
)
def test_run_record_reflects_the_new_evaluation(env, bucket, tags, expected):
    evaluate(env, FakeEvaluatorRunner(bucket=bucket, tags=tags))
    rec = load_run(env["run_dir"])
    assert rec.resolved is expected  # no migration needed


def test_results_cli_shows_pass_after_evaluation(env):
    from typer.testing import CliRunner

    from agent_harness_lab.cli.app import app

    args = ["results", "--raw-dir", str(env["raw"]), "--task", TASK, "--metrics-source", "none"]
    before = CliRunner().invoke(app, args, env={"COLUMNS": "220"}).output
    assert "N/A" in before and "PASS" not in before  # not evaluated yet
    evaluate(env, FakeEvaluatorRunner(bucket="resolved_ids"))
    after = CliRunner().invoke(app, args, env={"COLUMNS": "220"}).output
    assert "PASS" in after and "FAIL" not in after


def test_infra_failure_is_excluded_from_the_correctness_denominator(tmp_path, task_repo):
    from agent_harness_lab.reporting.compare import build_comparison
    from agent_harness_lab.results.service import load_runs

    raw = tmp_path / "raw"
    for name, bucket, tags in (
        ("a", "resolved_ids", ()),
        ("b", "unresolved_ids", ()),
        ("c", "unresolved_ids", ("infra_failure_ids",)),
    ):
        d = make_ahl_run(tmp_path / name, raw)
        runner = FakeEvaluatorRunner(bucket=bucket, tags=tags)
        run_evaluation(ADAPTER, make_ctx(d, task_repo, make_python(tmp_path / name), runner))
    mock = load_runs(raw).records
    assert sorted(str(r.resolved) for r in mock) == ["False", "None", "True"]
    other = [r.model_copy(update={"harness": "Other", "run_id": "other-1"}) for r in mock[:1]]
    cmp = build_comparison([*mock, *other], ["Mock", "Other"])
    summary = next(s for s in cmp.summaries if s.harness == "Mock")
    assert (summary.runs, summary.evaluated, summary.resolved) == (3, 2, 1)
    assert summary.success_rate == 0.5  # 1 of the 2 runs that got a verdict, not 1 of 3
    assert any("no valid evaluation" in w for w in cmp.warnings)


# ---- historical compatibility ----
def test_historical_evaluation_files_remain_readable(tmp_path):
    run = make_run(tmp_path, "h1", "pi-swebench-reference", outcome="resolved")
    assert load_evaluation(run, "h1") == (EvaluationState.RESOLVED, True)
    other = make_run(tmp_path, "h2", "opencode-swebench", outcome="unresolved", instanceId="p__b-2")
    assert load_evaluation(other, "h2") == (EvaluationState.UNRESOLVED, False)


def test_new_and_legacy_schemas_are_never_confused(env):
    evaluate(env)
    state, resolved = load_evaluation(env["run_dir"], env["run_dir"].name)
    assert (state, resolved) == (EvaluationState.RESOLVED, True)
    assert load_evaluation(env["run_dir"], "some-other-run") == (EvaluationState.INVALID, None)


def test_a_new_evaluation_json_is_not_mistaken_for_a_legacy_one(env):
    evaluate(env)
    data = json.loads((env["run_dir"] / "evaluation.json").read_text())
    assert (
        "schema_version" in data and "schemaVersion" not in data and "generationRunId" not in data
    )


# ---- harness independence and the evaluator-data boundary ----
def test_pi_and_opencode_style_runs_use_the_identical_evaluation_path(tmp_path, task_repo):
    raw = tmp_path / "raw"
    runners = {}
    for harness in ("mock-pi", "mock-opencode"):
        d = make_ahl_run(tmp_path / harness, raw, harness_name=harness)
        runner = FakeEvaluatorRunner()
        run_evaluation(ADAPTER, make_ctx(d, task_repo, make_python(tmp_path / harness), runner))
        call = next(c for c in runner.calls if "eval" in c["argv"])
        flags = [a for a in call["argv"][2:] if a.startswith("-")]
        runners[harness] = (flags, call["argv"][2:5])
    assert runners["mock-pi"] == runners["mock-opencode"]  # same command shape, same code path


def test_evaluation_package_has_no_harness_knowledge():
    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "agent_harness_lab" / "evaluation"
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("agent_harness_lab.harnesses"), path.name
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert node.value not in {"pi", "opencode", "Pi", "OpenCode"}, path.name


def test_evaluator_files_are_reachable_only_through_the_evaluation_layer(task_repo, monkeypatch):
    opened = []
    real = pathlib.Path.open

    def spy(self, *a, **k):
        opened.append(self.name)
        return real(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "open", spy)
    files = locate_evaluator_files(task_repo, TASK)
    assert files.complete and set(files.sha256) == set(HASHED_FILES)
    assert set(opened) <= {*HASHED_FILES, *BUILD_FILES}  # never gold.patch / hints.md
    assert set(files.build_sha256) == set(BUILD_FILES)  # Dockerfile + task.yaml hashed


def test_normal_task_api_still_cannot_read_evaluator_data(task_repo, monkeypatch):
    from agent_harness_lab.benchmarks.swebench_multilingual import SWEBenchMultilingual

    opened = []
    real = pathlib.Path.open

    def spy(self, *a, **k):
        opened.append(self.name)
        return real(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "open", spy)
    a = SWEBenchMultilingual(task_repo, FakeGit())
    a.list_tasks()
    a.get_task(TASK)
    assert set(opened) <= {"task.yaml", "problem_statement.md"}


def test_the_benchmark_adapter_never_imports_the_evaluator_data_module():
    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "agent_harness_lab" / "benchmarks"
    for name in ("swebench_multilingual.py", "base.py", "registry.py"):
        assert "evaluator_data" not in (src / name).read_text(), name


def test_missing_evaluator_task_files_are_reported_by_name(task_repo):
    (task_repo / "tasks" / TASK / "eval.sh").unlink()
    files = locate_evaluator_files(task_repo, TASK)
    assert files.missing == ["eval.sh"] and not files.complete and "eval.sh" not in files.sha256
