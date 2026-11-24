import json

import pytest
from typer.testing import CliRunner

from agent_harness_lab.cli.app import app
from agent_harness_lab.results.service import filter_runs, load_runs
from tests.factories import make_run

runner = CliRunner()
WIDE = {"COLUMNS": "220"}


@pytest.fixture
def raw(tmp_path):
    raw = tmp_path / "raw"
    make_run(raw, "t1-pi", "pi-swebench-reference", instanceId="proj__a-1")
    make_run(raw, "t1-oc", "opencode-swebench", instanceId="proj__a-1", outcome="unresolved")
    make_run(raw, "t2-pi", "pi-swebench-baseline", instanceId="proj__b-2")
    return raw


def invoke(raw, *args):
    return runner.invoke(app, ["results", "--raw-dir", str(raw), *args], env=WIDE)


def test_results_table_lists_runs(raw):
    result = invoke(raw)
    assert result.exit_code == 0
    for text in ("proj__a-1", "OpenCode", "gpt-6-astra", "PASS", "FAIL", "3 run(s)"):
        assert text in result.output


def test_results_filters(raw):
    records = load_runs(raw).records
    assert len(filter_runs(records, harnesses=["pi"])) == 2
    assert len(filter_runs(records, tasks=["a-1"])) == 2
    assert len(filter_runs(records, models=["GPT-6"])) == 2
    assert len(filter_runs(records, references=["opencode-frontier"])) == 1
    assert len(filter_runs(records, harnesses=["pi", "opencode"])) == 3  # repeated = OR
    assert len(filter_runs(records, harnesses=["pi"], tasks=["b-2"])) == 1  # different = AND
    assert filter_runs(records, references=["nope"]) == []


def test_results_cli_filter_options(raw):
    result = invoke(raw, "--harness", "Pi", "--task", "b-2")
    assert "1 run(s)" in result.output and "OpenCode" not in result.output


def test_results_json_is_clean_machine_readable(raw):
    result = invoke(raw, "--json", "--harness", "OpenCode")
    assert result.exit_code == 0
    data = json.loads(result.stdout)  # whole stdout must be JSON, no decoration
    assert len(data) == 1
    assert data[0]["harness"] == "OpenCode"
    assert data[0]["resolved"] is False
    assert data[0]["evaluation_state"] == "unresolved"
    assert data[0]["cost_usd"] is None


def test_results_missing_dir_is_a_message_not_a_traceback(tmp_path):
    result = runner.invoke(app, ["results", "--raw-dir", str(tmp_path / "nope")], env=WIDE)
    assert result.exit_code == 1
    assert "No results directory found" in result.output
    assert "Traceback" not in result.output


def test_unreadable_run_dirs_are_reported_on_stderr_not_in_json(raw):
    bad = raw / "swebench" / "proj__a-1" / "broken"
    bad.mkdir()
    result = invoke(raw, "--json")
    assert "skipped 1 unreadable" in result.output  # visible to the user...
    assert len(json.loads(result.stdout)) == 3  # ...but not mixed into stdout
