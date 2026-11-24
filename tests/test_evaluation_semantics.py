"""How the official swebench 5.0.2 report is interpreted (Part B).

Every case below was observed with the real evaluator (synthetic patches only) or read from its
source; see docs/evaluation.md. Key facts: the primary buckets are resolved / unresolved /
empty_patch / error / incomplete; ``infra_failure_ids`` and ``ambiguous_failure_ids`` are
advisory TAGS over unresolved|error; an unappliable patch is ONLY an ``error_ids`` entry whose
sole evidence is the log marker ``>>>>> Patch Apply Failed``.
"""

import pytest

from agent_harness_lab.artifacts.legacy import load_evaluation
from agent_harness_lab.artifacts.models import EvaluationState
from agent_harness_lab.evaluation.models import Classification, classify, resolved_for
from agent_harness_lab.evaluation.service import run_evaluation
from agent_harness_lab.evaluation.swebench_multilingual import (
    APPLY_PATCH_FAIL_MARKER,
    SWEBenchMultilingualEvaluator,
    classify_report,
)
from tests.eval_fixtures import FakeEvaluatorRunner, make_ahl_run, make_ctx, make_python, report_for

APPLY_FAILED_LOG = (
    f"INFO - Failed to apply patch\n{APPLY_PATCH_FAIL_MARKER}:\nerror: patch failed\n"
)
TIMEOUT_LOG = ">>>>> Applied Patch\nINFO - Test timed out after 5 seconds.\n"
ADAPTER = SWEBenchMultilingualEvaluator()


def classify_for(report, **kw):
    kw.setdefault("exit_code", 0)
    return classify_report(report, "t", timed_out=False, **kw)


# ---- model failures ----
def test_assertion_failure_is_a_plain_unresolved_benchmark_fail():
    state, official, warnings, error = classify_for(report_for("t", "unresolved_ids"))
    assert state is EvaluationState.UNRESOLVED and error is None and warnings == []
    assert resolved_for(state) is False


def test_test_command_failure_from_broken_code_is_unresolved_not_an_evaluator_error():
    """Observed: a patch with a syntax error makes the tests crash on import -> `unresolved`,
    completed, no tags; the instance report.json says no parseable test results."""
    inst = {"patch_is_None": False, "patch_exists": True, "patch_successfully_applied": False,
            "resolved": False, "infra_failure": False}  # fmt: skip
    state, official, _, _ = classify_for(report_for("t", "unresolved_ids"), instance_report=inst)
    assert state is EvaluationState.UNRESOLVED
    assert official["instance_report"] == inst  # the raw official entry is preserved


def test_unappliable_patch_is_a_model_failure_kept_in_the_denominator():
    """Observed: error_ids only, nothing tagged, failure_reasons empty, exit 0."""
    state, official, _, error = classify_for(
        report_for("t", "error_ids", completed=False), instance_log=APPLY_FAILED_LOG
    )
    assert state is EvaluationState.PATCH_APPLY_FAILED
    assert resolved_for(state) is False and classify(state) is Classification.BENCHMARK_FAIL
    assert "model" in error and official["log_markers"] == [APPLY_PATCH_FAIL_MARKER]


@pytest.mark.parametrize(
    "what", ["a text that is not a diff", "whitespace only", "diff of a missing file"]
)
def test_every_unappliable_patch_shape_lands_in_the_same_category(what):
    # all three were observed to produce the identical official report + log marker
    state, *_ = classify_for(
        report_for("t", "error_ids", completed=False), instance_log=APPLY_FAILED_LOG
    )
    assert state is EvaluationState.PATCH_APPLY_FAILED, what


def test_empty_patch_is_a_benchmark_fail():
    state, *_ = classify_for(report_for("t", "empty_patch_ids", completed=False))
    assert state is EvaluationState.EMPTY_PATCH and resolved_for(state) is False


# ---- infrastructure failures ----
@pytest.mark.parametrize("bucket", ["unresolved_ids", "error_ids"])
def test_environment_tier_signature_means_infrastructure_failure(bucket):
    r = report_for("t", bucket, tags=("infra_failure_ids",), reason="container_unavailable")
    state, official, _, error = classify_for(r)
    assert state is EvaluationState.INFRA_FAILURE and resolved_for(state) is None
    assert (
        official["failure_reason"] == "container_unavailable" and "container_unavailable" in error
    )
    assert classify(state) is Classification.EVALUATION_FAILURE


def test_evaluator_process_failure_with_no_report_is_infrastructure():
    """Observed: Docker unreachable -> exit code 1, no report file at all."""
    state, _, _, error = classify_for(None, exit_code=1)
    assert state is EvaluationState.INFRA_FAILURE and "no readable report" in error


def test_evaluator_process_timeout_is_infrastructure():
    state, *_ = classify_report(report_for("t", "resolved_ids"), "t", exit_code=0, timed_out=True)
    assert state is EvaluationState.INFRA_FAILURE


def test_error_with_no_recognised_cause_is_an_evaluator_error_without_verdict():
    """e.g. container creation / grading exception: error_ids, no marker, no tag."""
    log = "ERROR - Error in evaluating model for t: boom\n"
    state, _, _, error = classify_for(
        report_for("t", "error_ids", completed=False), instance_log=log
    )
    assert state is EvaluationState.ERROR and resolved_for(state) is None
    assert "no recognised cause" in error


# ---- genuinely ambiguous ----
def test_test_timeout_is_ambiguous_not_a_confirmed_model_or_infra_failure():
    """Observed: error_ids + ambiguous tag `tests_timed_out` + log 'Test timed out'."""
    r = report_for("t", "error_ids", tags=("ambiguous_failure_ids",), reason="tests_timed_out")
    state, official, _, _ = classify_for(r, instance_log=TIMEOUT_LOG)
    assert state is EvaluationState.AMBIGUOUS_FAILURE and resolved_for(state) is None
    assert official["failure_reason"] == "tests_timed_out" and official["ambiguous_failure_tag"]
    assert classify(state) is Classification.AMBIGUOUS


def test_timeout_marker_alone_is_ambiguous_too():
    state, *_ = classify_for(
        report_for("t", "error_ids", completed=False), instance_log=TIMEOUT_LOG
    )
    assert state is EvaluationState.AMBIGUOUS_FAILURE


def test_unresolved_with_an_ambiguous_signature_is_ambiguous():
    r = report_for("t", "unresolved_ids", tags=("ambiguous_failure_ids",), reason="missing_module")
    state, official, _, _ = classify_for(r)
    assert (
        state is EvaluationState.AMBIGUOUS_FAILURE
        and official["failure_reason"] == "missing_module"
    )


def test_unresolved_with_an_infra_signature_is_infrastructure():
    r = report_for("t", "unresolved_ids", tags=("infra_failure_ids",), reason="out_of_memory")
    state, *_ = classify_for(r)
    assert state is EvaluationState.INFRA_FAILURE


def test_apply_failure_marker_wins_over_nothing_else_evidence():
    # a patch-apply failure is decided by the official marker, not by guessing
    state, *_ = classify_for(
        report_for("t", "error_ids", completed=False), instance_log=APPLY_FAILED_LOG + TIMEOUT_LOG
    )
    assert state is EvaluationState.PATCH_APPLY_FAILED


# ---- the raw official entry is preserved ----
def test_raw_official_entry_failure_reason_and_markers_are_preserved():
    r = report_for("t", "error_ids", tags=("ambiguous_failure_ids",), reason="tests_timed_out")
    _, official, _, _ = classify_for(r, instance_log=TIMEOUT_LOG, instance_report=None)
    assert set(official) >= {
        "buckets", "infra_failure_tag", "ambiguous_failure_tag", "completed", "submitted",
        "report_schema_version", "instance_report", "log_markers", "failure_reason",
    }  # fmt: skip
    assert official["buckets"] == ["error"] and official["report_schema_version"] == 2


# ---- correctness denominators ----
def test_patch_apply_failure_stays_in_the_denominator_but_infra_does_not(tmp_path, task_repo):
    from agent_harness_lab.reporting.compare import build_comparison
    from agent_harness_lab.results.service import load_runs

    raw = tmp_path / "raw"
    cases = {
        "ok": dict(bucket="resolved_ids"),
        "apply": dict(bucket="error_ids", log_text=APPLY_FAILED_LOG),
        "infra": dict(bucket="unresolved_ids", tags=("infra_failure_ids",), reason="out_of_memory"),
        "ambiguous": dict(
            bucket="error_ids", tags=("ambiguous_failure_ids",), reason="tests_timed_out"
        ),
    }
    for name, kw in cases.items():
        d = make_ahl_run(tmp_path / name, raw)
        run_evaluation(
            ADAPTER, make_ctx(d, task_repo, make_python(tmp_path / name), FakeEvaluatorRunner(**kw))
        )
    mock = load_runs(raw).records
    assert sorted(str(r.resolved) for r in mock) == ["False", "None", "None", "True"]
    other = [r.model_copy(update={"harness": "Other", "run_id": "o"}) for r in mock[:1]]
    summary = next(
        s
        for s in build_comparison([*mock, *other], ["Mock", "Other"]).summaries
        if s.harness == "Mock"
    )
    assert (summary.runs, summary.evaluated, summary.resolved) == (4, 2, 1)
    assert summary.success_rate == 0.5  # 1 pass, 1 apply-failure; infra + ambiguous excluded


@pytest.mark.parametrize(
    ("kw", "state", "resolved", "cls"),
    [
        (
            dict(bucket="error_ids", log_text=APPLY_FAILED_LOG),
            "patch_apply_failed", False, "benchmark_fail",
        ),
        (dict(bucket="error_ids", tags=("ambiguous_failure_ids",), reason="tests_timed_out"),
         "ambiguous_failure", None, "ambiguous"),
        (
            dict(bucket="unresolved_ids", tags=("infra_failure_ids",)),
            "infra_failure", None, "evaluation_failure",
        ),
        (dict(bucket="error_ids"), "error", None, "evaluation_failure"),
    ],
)  # fmt: skip
def test_end_to_end_documents_for_each_category(tmp_path, task_repo, kw, state, resolved, cls):
    d = make_ahl_run(tmp_path, tmp_path / "raw")
    ctx = make_ctx(d, task_repo, make_python(tmp_path), FakeEvaluatorRunner(**kw))
    _, doc = run_evaluation(ADAPTER, ctx)
    assert (doc.state.value, doc.resolved, doc.classification.value) == (state, resolved, cls)
    assert doc.official_result["buckets"]  # the raw official entry is in evaluation.json
    assert load_evaluation(d, d.name) == (EvaluationState(state), resolved)


def test_patch_apply_failed_state_is_readable_from_evaluation_json(tmp_path, task_repo):
    d = make_ahl_run(tmp_path, tmp_path / "raw")
    ctx = make_ctx(
        d,
        task_repo,
        make_python(tmp_path),
        FakeEvaluatorRunner(bucket="error_ids", log_text=APPLY_FAILED_LOG),
    )
    run_evaluation(ADAPTER, ctx)
    from agent_harness_lab.artifacts.legacy import load_run

    rec = load_run(d)
    assert rec.evaluation_state is EvaluationState.PATCH_APPLY_FAILED and rec.resolved is False


# ---- no behaviour change for the verdicts that were already right ----
@pytest.mark.parametrize(
    ("bucket", "state", "resolved"),
    [
        ("resolved_ids", EvaluationState.RESOLVED, True),
        ("unresolved_ids", EvaluationState.UNRESOLVED, False),
        ("empty_patch_ids", EvaluationState.EMPTY_PATCH, False),
    ],
)
def test_official_pass_fail_parsing_is_unchanged(bucket, state, resolved):
    got, *_ = classify_for(report_for("t", bucket, completed=bucket != "empty_patch_ids"))
    assert got is state and resolved_for(got) is resolved
