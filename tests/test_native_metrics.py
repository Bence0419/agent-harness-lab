import json

import pytest
from typer.testing import CliRunner

from agent_harness_lab.artifacts.legacy import load_run
from agent_harness_lab.cli import results as results_cli
from agent_harness_lab.cli.app import app
from agent_harness_lab.metrics.patch import PatchStatError
from agent_harness_lab.results import native_metrics
from agent_harness_lab.results.native_metrics import NativeTraceMetrics
from agent_harness_lab.results.service import (
    MetricsSourceName,
    attach_metrics,
    build_metrics_source,
    filter_runs,
    load_runs,
)
from tests.factories import make_run
from tests.test_native_helpers import FakeNumstat
from tests.trace_fixtures import jl, oc_finish, oc_tool, pi_end, pi_start, pi_turn, write_trace

runner = CliRunner()
WIDE = {"COLUMNS": "220"}


def pi_run(tmp_path, run_id="pi1", **kw):
    return make_run(tmp_path, run_id, "pi-swebench-reference", **kw)


def oc_run(tmp_path, run_id="oc1", **kw):
    return make_run(tmp_path, run_id, "opencode-swebench", **kw)


PI_TRACE = jl(
    pi_turn(inp=3, out=45, cache_write=1351, cost=0.25),
    pi_start("bash", command="pytest -q"),
    pi_end("bash", is_error=True),
    pi_start("read", path="a.py"),
    pi_end("read"),
    pi_turn(inp=2, out=5, cache_read=100, cache_write=0, cost=0.5),
)

OC_TRACE = jl(
    oc_tool("shell", command="npm test"),
    oc_tool("grep", status="error", pattern="x"),
    oc_tool("patch", path="a.js"),
    oc_finish(inp=3, out=23, read=9, write=4565, cost=0.25),
)


def enrich(run_dir, probe=None):
    rec = load_run(run_dir)
    out, warnings = NativeTraceMetrics(probe or FakeNumstat()).enrich([rec])
    return out[0], warnings


def test_native_source_pi_metrics(tmp_path):
    run = pi_run(tmp_path)
    write_trace(run, PI_TRACE)
    rec, warnings = enrich(run)
    assert warnings == []
    assert (rec.model_turns, rec.tool_calls, rec.tool_errors) == (2, 2, 1)
    assert (rec.bash_calls, rec.read_calls, rec.verification_calls) == (1, 1, 1)
    assert (rec.input_tokens, rec.output_tokens, rec.cache_read_tokens) == (5, 50, 100)
    assert rec.cache_write_tokens == 1351
    assert rec.cost_usd == 0.75


def test_native_source_opencode_metrics(tmp_path):
    run = oc_run(tmp_path)
    write_trace(run, OC_TRACE)
    rec, _ = enrich(run)
    assert (rec.model_turns, rec.tool_calls, rec.tool_errors) == (1, 3, 1)
    assert (rec.bash_calls, rec.edit_calls, rec.other_tool_calls) == (1, 1, 1)
    assert rec.verification_calls == 1
    assert (rec.input_tokens, rec.output_tokens) == (3, 23)
    assert (rec.cache_read_tokens, rec.cache_write_tokens) == (9, 4565)
    assert rec.cost_usd == 0.25


def test_native_source_records_metric_schema_and_source(tmp_path):
    run = pi_run(tmp_path)
    write_trace(run, PI_TRACE)
    rec, _ = enrich(run)
    assert rec.metrics_schema == "ahl-metrics-v1"
    assert rec.metrics_source == "native-trace"


def test_harness_comes_from_metadata_not_trace_content_or_names(tmp_path):
    # OpenCode-shaped trace content in a run whose metadata says Pi -> Pi parser, no tool_use
    run = pi_run(tmp_path, "run-named-opencode-lookalike")
    write_trace(run, OC_TRACE)
    rec, _ = enrich(run)
    assert rec.tool_calls == 0 and rec.model_turns == 0  # genuine zeros: valid trace, no Pi events


def test_missing_trace_gives_unavailable_metrics_not_zeros(tmp_path):
    rec, warnings = enrich(pi_run(tmp_path))
    for f in ("model_turns", "tool_calls", "tool_errors", "verification_calls", "cost_usd",
              "input_tokens", "read_calls", "metrics_schema"):  # fmt: skip
        assert getattr(rec, f) is None, f
    assert any("trace.jsonl missing" in w for w in warnings)


def test_empty_trace_unavailable_and_distinct_from_zero_trace(tmp_path):
    empty = pi_run(tmp_path, "empty")
    write_trace(empty, "")
    zero = pi_run(tmp_path, "zero", instanceId="proj__other-2")
    write_trace(zero, jl({"type": "agent_start"}))
    rec_empty, warn = enrich(empty)
    rec_zero, _ = enrich(zero)
    assert rec_empty.tool_calls is None and any("empty trace" in w for w in warn)
    assert rec_zero.tool_calls == 0 and rec_zero.cost_usd == 0.0


def test_malformed_and_truncated_lines_warn_but_metrics_survive(tmp_path):
    run = pi_run(tmp_path)
    write_trace(run, PI_TRACE + "{broken\n" + '{"type": "tool_exec')
    rec, warnings = enrich(run)
    assert rec.tool_calls == 2 and rec.model_turns == 2
    text = " ".join(warnings)
    assert "1 malformed line(s) skipped" in text and "truncated final line" in text


def test_all_garbage_trace_is_unavailable(tmp_path):
    run = pi_run(tmp_path)
    write_trace(run, "garbage\nmore garbage\n")
    rec, warnings = enrich(run)
    assert rec.tool_calls is None
    assert any("trace metrics unavailable" in w for w in warnings)


def test_patch_metrics_attached_via_fake_probe(tmp_path):
    run = pi_run(tmp_path)
    write_trace(run, PI_TRACE)
    (run / "model.patch").write_text("diff")
    rec, _ = enrich(run, FakeNumstat("5\t1\tsrc/a.js\n2\t0\ttest/a.test.js\n"))
    assert (rec.files_changed, rec.test_files_changed, rec.non_test_files_changed) == (2, 1, 1)
    assert (rec.insertions, rec.deletions) == (7, 1)


def test_missing_patch_leaves_patch_metrics_none(tmp_path):
    run = pi_run(tmp_path)
    write_trace(run, PI_TRACE)
    rec, _ = enrich(run)
    assert rec.files_changed is None and rec.insertions is None
    assert rec.tool_calls == 2  # trace metrics unaffected


def test_patch_probe_failure_warns_and_keeps_trace_metrics(tmp_path):
    run = pi_run(tmp_path)
    write_trace(run, PI_TRACE)
    (run / "model.patch").write_text("diff")
    rec, warnings = enrich(run, FakeNumstat(error=PatchStatError("boom")))
    assert rec.files_changed is None and rec.tool_calls == 2
    assert any("patch metrics unavailable" in w for w in warnings)


def test_patch_metrics_survive_a_missing_trace(tmp_path):
    run = pi_run(tmp_path)
    (run / "model.patch").write_text("diff")
    rec, _ = enrich(run, FakeNumstat("1\t0\tsrc/a.js\n"))
    assert rec.files_changed == 1 and rec.tool_calls is None
    assert rec.metrics_schema == "ahl-metrics-v1"  # patch stats are v1 metrics too


def test_unsupported_harness_warns_and_leaves_metrics_none(tmp_path):
    run = pi_run(tmp_path)
    write_trace(run, PI_TRACE)
    rec = load_run(run).model_copy(update={"harness": "Mystery"})
    out, warnings = NativeTraceMetrics(FakeNumstat()).enrich([rec])
    assert out[0].tool_calls is None
    assert any("unsupported harness" in w for w in warnings)


def test_extra_agent_pass_trace_is_flagged_not_silently_ignored(tmp_path):
    run = make_run(tmp_path, "v1", "pi-swebench-verification")
    write_trace(run, PI_TRACE)
    (run / "verification-trace.jsonl").write_text(PI_TRACE)
    rec, warnings = enrich(run)
    assert rec.tool_calls == 2  # solve pass only, exactly as the v1 extractor
    assert any("verification-trace.jsonl" in w and "not included" in w for w in warnings)


def test_native_source_never_modifies_artifacts(tmp_path):
    run = pi_run(tmp_path)
    write_trace(run, PI_TRACE)
    before = {p.name: p.read_bytes() for p in run.iterdir()}
    enrich(run)
    assert {p.name: p.read_bytes() for p in run.iterdir()} == before


# ---- lazy parsing: filter first, parse only selected runs ---------------------
@pytest.fixture
def two_runs(tmp_path):
    raw = tmp_path / "raw"
    a = make_run(raw, "pi-a", "pi-swebench-reference", instanceId="p__a-1")
    b = make_run(raw, "oc-b", "opencode-swebench", instanceId="p__b-2")
    write_trace(a, PI_TRACE)
    write_trace(b, OC_TRACE)
    return raw


def test_load_runs_never_reads_traces(two_runs, monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("trace read during lightweight load")

    monkeypatch.setattr(native_metrics, "read_events", forbidden)
    assert len(load_runs(two_runs).records) == 2


def test_only_filtered_runs_are_parsed(two_runs, monkeypatch):
    opened = []
    real = native_metrics.read_events

    def spy(path, harness, report):
        opened.append(path.parent.name)
        return real(path, harness, report)

    monkeypatch.setattr(native_metrics, "read_events", spy)
    records = filter_runs(load_runs(two_runs).records, harnesses=["Pi"])
    attach_metrics(records, NativeTraceMetrics(FakeNumstat()))
    assert opened == ["pi-a"]  # the OpenCode trace was never opened


def test_cli_filters_before_parsing(two_runs, monkeypatch):
    seen = []

    class Spy:
        def enrich(self, records):
            seen.extend(r.run_id for r in records)
            return records, []

    monkeypatch.setattr(results_cli, "build_metrics_source", lambda name, raw: Spy())
    result = runner.invoke(
        app, ["results", "--raw-dir", str(two_runs), "--harness", "OpenCode"], env=WIDE
    )
    assert result.exit_code == 0 and seen == ["oc-b"]


def test_cli_compare_filters_harness_before_parsing(two_runs, monkeypatch):
    make_run(two_runs, "mystery", "pi-swebench-baseline", instanceId="p__b-2")
    seen = []

    class Spy:
        def enrich(self, records):
            seen.extend(r.run_id for r in records)
            return records, []

    monkeypatch.setattr(results_cli, "build_metrics_source", lambda name, raw: Spy())
    runner.invoke(
        app,
        ["compare", "--raw-dir", str(two_runs), "--harness", "Pi", "--harness", "OpenCode",
         "--task", "b-2"],
        env=WIDE,
    )  # fmt: skip
    assert sorted(seen) == ["mystery", "oc-b"]  # 'pi-a' (task a-1) excluded before parsing


def test_metrics_source_none_skips_traces(two_runs, monkeypatch):
    assert build_metrics_source(MetricsSourceName.NONE, two_runs) is None
    monkeypatch.setattr(
        native_metrics, "read_events", lambda *a, **k: (_ for _ in ()).throw(AssertionError())
    )
    result = runner.invoke(
        app, ["results", "--raw-dir", str(two_runs), "--metrics-source", "none", "--json"], env=WIDE
    )
    data = json.loads(result.stdout)
    assert result.exit_code == 0 and all(r["tool_calls"] is None for r in data)


def test_cli_default_is_native_and_shows_cost(two_runs):
    result = runner.invoke(app, ["results", "--raw-dir", str(two_runs), "--json"], env=WIDE)
    by_id = {r["run_id"]: r for r in json.loads(result.stdout)}
    assert by_id["pi-a"]["cost_usd"] == 0.75 and by_id["pi-a"]["metrics_schema"] == "ahl-metrics-v1"
    assert by_id["oc-b"]["tool_calls"] == 3


def test_cli_compare_uses_native_metrics_by_default(two_runs):
    make_run(two_runs, "oc-a", "opencode-swebench", instanceId="p__a-1")
    write_trace(two_runs / "swebench" / "p__a-1" / "oc-a", OC_TRACE)
    make_run(two_runs, "pi-b", "pi-swebench-reference", instanceId="p__b-2")
    write_trace(two_runs / "swebench" / "p__b-2" / "pi-b", PI_TRACE)
    result = runner.invoke(
        app,
        ["compare", "--raw-dir", str(two_runs), "--harness", "Pi", "--harness", "OpenCode"],
        env=WIDE,
    )
    line = next(ln for ln in result.output.splitlines() if "Tool calls" in ln)
    assert result.exit_code == 0 and "4" in line and "6" in line  # Pi 2+2, OpenCode 3+3
