import hashlib
import json
import os
from datetime import UTC, datetime, timedelta

import pytest
from typer.testing import CliRunner

from agent_harness_lab.artifacts.discovery import discover_runs
from agent_harness_lab.artifacts.hashing import sha256_file, sha256_text
from agent_harness_lab.artifacts.legacy import LegacyFormatError, detect_format, load_run
from agent_harness_lab.artifacts.run_record import COMPAT_METADATA_SCHEMA, compat_metadata
from agent_harness_lab.artifacts.writer import write_json_atomic, write_text_atomic
from agent_harness_lab.cli.app import app
from agent_harness_lab.execution.executor import RunRequest, execute_run
from agent_harness_lab.results.service import filter_runs, load_runs
from tests.factories import evaluation, make_run
from tests.fake_docker import FakeDocker
from tests.mock_harness import mock_plan, mock_provisioner, write_mock_harness

runner = CliRunner()
WIDE = {"COLUMNS": "200"}


def do_run(tmp_path, raw=None, **plan_kw):
    raw = raw or tmp_path / "raw"
    ticks = iter(
        datetime(2026, 11, 24, 10, 0, 0, tzinfo=UTC) + timedelta(seconds=i) for i in range(500)
    )
    plan = mock_plan(prompt="p", **plan_kw)
    request = RunRequest(
        plan=plan,
        prompt="p",
        provisioner=mock_provisioner(write_mock_harness(tmp_path)),
        raw_dir=raw,
    )
    doc = execute_run(request, docker=FakeDocker(), now=lambda: next(ticks), sleep=lambda s: None)
    return doc, raw / "swebench" / plan.task.instance_id / doc.run_id


# ---- hashing / atomic writing ----
def test_sha256_text_and_file(tmp_path):
    assert sha256_text("abc") == hashlib.sha256(b"abc").hexdigest()
    big = tmp_path / "big"
    big.write_bytes(os.urandom(3 * 1024 * 1024 + 17))  # larger than the streaming chunk
    assert sha256_file(big) == hashlib.sha256(big.read_bytes()).hexdigest()


def test_atomic_write_replaces_without_leaving_temp_files(tmp_path):
    p = tmp_path / "run.json"
    write_json_atomic(p, {"a": 1})
    write_json_atomic(p, {"a": 2})
    assert json.loads(p.read_text()) == {"a": 2}
    assert [f.name for f in tmp_path.iterdir()] == ["run.json"]


def test_atomic_write_failure_keeps_the_previous_file_intact(tmp_path, monkeypatch):
    p = tmp_path / "run.json"
    write_text_atomic(p, "original")

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        write_text_atomic(p, "new content")
    assert p.read_text() == "original"
    assert [f.name for f in tmp_path.iterdir()] == ["run.json"]  # temp file cleaned up


# ---- compat metadata ----
def test_compat_metadata_has_its_own_schema_and_is_not_a_legacy_shape(tmp_path):
    doc, d = do_run(tmp_path)
    meta = json.loads((d / "metadata.json").read_text())
    assert meta["schema"] == COMPAT_METADATA_SCHEMA == "ahl-metadata-v1"
    assert meta["runId"] == doc.run_id and meta["instanceId"] == "mock__task-1"
    assert meta["baseCommit"] == "a" * 40 and meta["state"] == "completed"
    assert meta == compat_metadata(doc)
    with pytest.raises(LegacyFormatError):  # it must not masquerade as a historical shape
        detect_format(meta)


# ---- discovery and precedence ----
def test_new_run_is_discovered_without_any_migration(tmp_path):
    doc, d = do_run(tmp_path)
    found = discover_runs(tmp_path / "raw")
    assert found.issues == []
    (rec,) = found.records
    assert rec.run_id == doc.run_id and rec.task == "mock__task-1"
    assert (rec.harness, rec.model, rec.provider) == ("Mock", "mock-1", "mock")
    assert rec.legacy_format == "ahl-run-v1" and rec.execution_state == "completed"
    assert rec.base_commit == "a" * 40 and rec.docker_image == "ahl-test-synthetic:1"
    assert rec.harness_commit == "c" * 40 and rec.harness_dirty is False
    assert rec.network_policy == "unrestricted" and rec.duration_seconds == 1
    assert rec.resolved is None and rec.evaluation_state.value == "not_evaluated"
    assert rec.artifact_dir == d


def test_run_json_is_preferred_over_metadata_json(tmp_path):
    doc, d = do_run(tmp_path)
    # overwrite metadata.json with a plausible LEGACY shape describing a different task
    legacy = {
        "runId": doc.run_id, "instanceId": "legacy__other-9", "piExitCode": 0, "piVersion": "1.0.2",
        "provider": "openai", "model": "gpt-5.4-mini", "thinking": "medium",
        "baseCommit": "b" * 40, "image": "x", "harnessCommit": "d" * 40, "harnessDirty": True,
        "taskRepoCommit": "e" * 40, "startedAt": "2026-01-01T00:00:00Z",
        "finishedAt": "2026-01-01T00:00:01Z",
    }  # fmt: skip
    (d / "metadata.json").write_text(json.dumps(legacy))
    rec = load_run(d)
    assert (
        rec.task == "mock__task-1" and rec.model == "mock-1" and rec.legacy_format == "ahl-run-v1"
    )


def test_compat_metadata_without_run_json_is_an_explicit_error(tmp_path):
    doc, d = do_run(tmp_path)
    (d / "run.json").unlink()
    with pytest.raises(LegacyFormatError, match="run.json is the canonical record"):
        load_run(d)
    assert discover_runs(tmp_path / "raw").issues[0].reason.startswith("ahl-metadata-v1")


@pytest.mark.parametrize("content", ["{not json", '{"schema_version": "ahl-run-v9"}', "[]"])
def test_malformed_or_unsupported_run_json_is_reported_not_crashed(tmp_path, content):
    _, d = do_run(tmp_path)
    (d / "run.json").write_text(content)
    found = discover_runs(tmp_path / "raw")
    assert found.records == [] and "run.json" in found.issues[0].reason


def test_evaluation_json_next_to_a_new_run_is_picked_up(tmp_path):
    doc, d = do_run(tmp_path)
    (d / "evaluation.json").write_text(json.dumps(evaluation(doc.run_id, "mock__task-1")))
    assert load_run(d).resolved is True


def test_legacy_runs_remain_readable_alongside_new_ones(tmp_path):
    raw = tmp_path / "raw"
    make_run(raw, "legacy-pi", "pi-swebench-reference", instanceId="proj__a-1")
    make_run(raw, "legacy-oc", "opencode-swebench", instanceId="proj__a-1")
    doc, _ = do_run(tmp_path, raw=raw)
    records = load_runs(raw).records
    assert {r.legacy_format for r in records} == {
        "pi-swebench-reference", "opencode-swebench", "ahl-run-v1",
    }  # fmt: skip
    assert len(filter_runs(records, harnesses=["mock"])) == 1
    legacy = next(r for r in records if r.run_id == "legacy-pi")
    assert legacy.execution_state is None  # historical runs have none


def test_ahl_results_cli_shows_a_new_run(tmp_path):
    do_run(tmp_path)
    r = runner.invoke(
        app,
        ["results", "--raw-dir", str(tmp_path / "raw"), "--metrics-source", "none", "--json"],
        env=WIDE,
    )
    assert r.exit_code == 0
    (rec,) = json.loads(r.stdout)
    assert rec["legacy_format"] == "ahl-run-v1" and rec["execution_state"] == "completed"
    table = runner.invoke(
        app, ["results", "--raw-dir", str(tmp_path / "raw"), "--metrics-source", "none"], env=WIDE
    )
    assert "mock__task-1" in table.output and "Mock" in table.output


def test_native_metrics_for_an_unsupported_harness_warn_instead_of_crashing(tmp_path):
    do_run(tmp_path)
    r = runner.invoke(app, ["results", "--raw-dir", str(tmp_path / "raw"), "--json"], env=WIDE)
    assert r.exit_code == 0 and "unsupported harness" in r.output
    assert json.loads(r.stdout)[0]["tool_calls"] is None  # unavailable, not zero


# ---- structural isolation of evaluator data ----
def test_the_plan_cannot_carry_a_task_directory_or_evaluator_paths(tmp_path):
    from agent_harness_lab.runplan.models import TaskRef

    assert "task_directory" not in TaskRef.model_fields
    doc, d = do_run(tmp_path)
    blob = (d / "run.json").read_text()
    for forbidden in ("gold.patch", "hints.md", "test.patch", "tests.json", "eval.sh"):
        assert forbidden not in blob
