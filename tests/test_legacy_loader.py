import json

import pytest

from agent_harness_lab.artifacts.discovery import discover_runs
from agent_harness_lab.artifacts.legacy import LegacyFormatError, detect_format, load_run
from agent_harness_lab.artifacts.models import EvaluationState
from tests.factories import SHAPES, make_run


@pytest.mark.parametrize("shape", list(SHAPES))
def test_detect_format_for_every_historical_shape(shape):
    assert detect_format(SHAPES[shape]) == shape


def test_pi_baseline_has_no_reference(tmp_path):
    rec = load_run(make_run(tmp_path, "r1", "pi-swebench-baseline"))
    assert (rec.harness, rec.harness_version) == ("Pi", "1.0.2")
    assert rec.reference is None
    assert rec.reasoning == "medium"
    assert rec.legacy_format == "pi-swebench-baseline"
    assert rec.network_policy is None


def test_pi_reference_shape(tmp_path):
    rec = load_run(make_run(tmp_path, "r1", "pi-swebench-reference"))
    assert rec.reference == "pi-frontier-astra-v1"
    assert rec.model == "gpt-6-astra" and rec.provider == "openai"
    assert rec.duration_seconds == 150
    assert rec.harness_commit == "c" * 40 and rec.harness_dirty is False
    assert rec.task_repo_commit.startswith("6e08cbc")
    assert rec.base_commit == "b" * 40
    assert rec.docker_image.startswith("swebench/")
    assert rec.started_at.isoformat() == "2026-11-18T10:00:00+00:00"


def test_verification_shape_uses_experiment_as_reference(tmp_path):
    rec = load_run(make_run(tmp_path, "r1", "pi-swebench-verification"))
    assert rec.legacy_format == "pi-swebench-verification"
    assert rec.reference == "verification-v1"
    assert rec.harness == "Pi"


def test_opencode_shape(tmp_path):
    rec = load_run(make_run(tmp_path, "r1", "opencode-swebench"))
    assert (rec.harness, rec.harness_version) == ("OpenCode", "2.2.0")
    assert rec.reasoning == "medium"  # from `variant`
    assert rec.network_policy == "openai-only-squid-v1"


def test_harness_comes_from_fields_not_directory_name(tmp_path):
    # Run id says "pi" but the structured fields say OpenCode.
    rec = load_run(make_run(tmp_path, "proj__repo-1-pi-lookalike", "opencode-swebench"))
    assert rec.harness == "OpenCode"


def test_missing_optional_fields_stay_none(tmp_path):
    run = make_run(
        tmp_path,
        "r1",
        "pi-swebench-baseline",
        remove=("harnessCommit", "harnessDirty", "image", "piVersion", "finishedAt", "model"),
    )
    rec = load_run(run)
    for name in (
        "harness_commit",
        "harness_dirty",
        "docker_image",
        "harness_version",
        "duration_seconds",
        "model",
        "cost_usd",
        "tool_calls",
        "model_turns",
    ):
        assert getattr(rec, name) is None, name


def test_pass_evaluation(tmp_path):
    rec = load_run(make_run(tmp_path, "r1", outcome="resolved"))
    assert rec.resolved is True
    assert rec.evaluation_state is EvaluationState.RESOLVED


@pytest.mark.parametrize(
    ("outcome", "state"),
    [
        ("unresolved", EvaluationState.UNRESOLVED),
        ("emptyPatch", EvaluationState.EMPTY_PATCH),
    ],
)
def test_benchmark_fail_evaluations_are_resolved_false(tmp_path, outcome, state):
    rec = load_run(make_run(tmp_path, "r1", outcome=outcome))
    assert rec.resolved is False
    assert rec.evaluation_state is state


@pytest.mark.parametrize(
    ("outcome", "state"),
    [
        ("infraFailure", EvaluationState.INFRA_FAILURE),
        ("error", EvaluationState.ERROR),
        ("ambiguousFailure", EvaluationState.AMBIGUOUS_FAILURE),
        ("incomplete", EvaluationState.INCOMPLETE),
    ],
)
def test_non_verdict_evaluations_are_not_benchmark_failures(tmp_path, outcome, state):
    rec = load_run(make_run(tmp_path, "r1", outcome=outcome))
    assert rec.resolved is None  # no verdict: excluded from correctness denominators
    assert rec.evaluation_state is state


def test_infra_failure_is_neither_pass_nor_fail(tmp_path):
    assert load_run(make_run(tmp_path, "r1", outcome="infraFailure")).resolved is None


def test_missing_evaluation_is_unknown_not_fail(tmp_path):
    rec = load_run(make_run(tmp_path, "r1", outcome=None))
    assert rec.resolved is None
    assert rec.evaluation_state is EvaluationState.NOT_EVALUATED


def test_malformed_evaluation_is_invalid(tmp_path):
    run = make_run(tmp_path, "r1")
    (run / "evaluation.json").write_text("{nope")
    rec = load_run(run)
    assert rec.resolved is None and rec.evaluation_state is EvaluationState.INVALID


def test_evaluation_for_another_run_is_invalid(tmp_path):
    run = make_run(tmp_path, "r1")
    data = json.loads((run / "evaluation.json").read_text())
    data["generationRunId"] = "some-other-run"
    (run / "evaluation.json").write_text(json.dumps(data))
    assert load_run(run).evaluation_state is EvaluationState.INVALID


def test_malformed_metadata_raises(tmp_path):
    run = make_run(tmp_path, "r1")
    (run / "metadata.json").write_text("{not json")
    with pytest.raises(LegacyFormatError, match="malformed"):
        load_run(run)


def test_metadata_not_an_object_raises(tmp_path):
    run = make_run(tmp_path, "r1")
    (run / "metadata.json").write_text("[1, 2]")
    with pytest.raises(LegacyFormatError, match="not a JSON object"):
        load_run(run)


def test_unrecognized_shape_raises(tmp_path):
    run = make_run(tmp_path, "r1")
    (run / "metadata.json").write_text(json.dumps({"runId": "r1", "instanceId": "x"}))
    with pytest.raises(LegacyFormatError, match="unrecognized"):
        load_run(run)


def test_missing_run_id_raises(tmp_path):
    run = make_run(tmp_path, "r1")
    meta = json.loads((run / "metadata.json").read_text())
    del meta["runId"]
    (run / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(LegacyFormatError, match="runId"):
        load_run(run)


def test_directory_without_metadata_is_reported(tmp_path):
    good = make_run(tmp_path, "r1")
    broken = good.parent / "proj__repo-1-incomplete"
    broken.mkdir()
    (broken / "trace.jsonl").write_text("")
    found = discover_runs(tmp_path)
    assert [r.run_id for r in found.records] == ["r1"]
    assert len(found.issues) == 1
    assert "no metadata.json" in found.issues[0].reason


def test_loader_never_modifies_artifacts(tmp_path):
    run = make_run(tmp_path, "r1")
    before = {p.name: p.read_bytes() for p in run.iterdir()}
    load_run(run)
    assert {p.name: p.read_bytes() for p in run.iterdir()} == before
