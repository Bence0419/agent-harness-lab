import pytest
from typer.testing import CliRunner

from agent_harness_lab.cli.app import app
from agent_harness_lab.reporting.compare import NoComparableRuns, build_comparison
from agent_harness_lab.results.service import load_runs
from tests.factories import make_run

runner = CliRunner()
WIDE = {"COLUMNS": "220"}


def metrics(**kw):
    return kw


def build(tmp_path, runs):
    raw = tmp_path / "raw"
    for run_id, shape, kw in runs:
        make_run(raw, run_id, shape, **kw)
    return raw, load_runs(raw).records


def two_harness_runs(**oc_overrides):
    return [
        ("pi-a", "pi-swebench-reference", {"instanceId": "p__a-1"}),
        ("pi-b", "pi-swebench-reference", {"instanceId": "p__b-2"}),
        ("oc-a", "opencode-swebench", {"instanceId": "p__a-1", **oc_overrides}),
        (
            "oc-b",
            "opencode-swebench",
            {"instanceId": "p__b-2", "outcome": "unresolved", **oc_overrides},
        ),
    ]


def test_compare_two_harnesses_basic(tmp_path):
    _, records = build(tmp_path, two_harness_runs())
    c = build_comparison(records, ["Pi", "OpenCode"])
    pi, oc = c.summaries
    assert (pi.harness, pi.resolved, pi.evaluated, pi.success_rate) == ("Pi", 2, 2, 1.0)
    assert (oc.resolved, oc.evaluated, oc.success_rate) == (1, 2, 0.5)
    assert pi.totals["duration_seconds"] == 300  # 2 runs x 150 s
    assert c.tasks_compared == ["p__a-1", "p__b-2"]


def test_missing_metrics_are_none_not_zero(tmp_path):
    _, records = build(tmp_path, two_harness_runs())
    c = build_comparison(records, ["Pi", "OpenCode"])
    for s in c.summaries:
        assert s.totals["tool_calls"] is None
        assert s.totals["cost_usd"] is None


def test_partial_metrics_do_not_produce_partial_sums(tmp_path):
    _, records = build(tmp_path, two_harness_runs())
    records[0] = records[0].model_copy(update={"tool_calls": 5})  # only one run has it
    c = build_comparison(records, ["Pi", "OpenCode"])
    assert all(s.totals["tool_calls"] is None for s in c.summaries)


def test_real_zero_is_kept_as_zero(tmp_path):
    _, records = build(tmp_path, two_harness_runs())
    records = [r.model_copy(update={"verification_calls": 0}) for r in records]
    c = build_comparison(records, ["Pi", "OpenCode"])
    assert c.summaries[0].totals["verification_calls"] == 0


def test_only_common_tasks_are_compared(tmp_path):
    runs = two_harness_runs() + [("pi-c", "pi-swebench-reference", {"instanceId": "p__c-3"})]
    _, records = build(tmp_path, runs)
    c = build_comparison(records, ["Pi", "OpenCode"])
    assert c.tasks_compared == ["p__a-1", "p__b-2"]
    assert any("1 task(s) not run by every harness" in w for w in c.warnings)


def test_mixed_network_policy_warning(tmp_path):
    _, records = build(tmp_path, two_harness_runs())  # Pi: not recorded, OpenCode: recorded
    c = build_comparison(records, ["Pi", "OpenCode"])
    warning = next(w for w in c.warnings if "mixed network policies" in w)
    assert "not recorded (2)" in warning and "openai-only-squid-v1 (2)" in warning


def test_uniform_network_policy_has_no_network_warning(tmp_path):
    runs = two_harness_runs(networkPolicy="p1")
    runs = [(i, s, {**kw, "networkPolicy": "p1"}) for i, s, kw in runs]
    _, records = build(tmp_path, runs)
    c = build_comparison(records, ["Pi", "OpenCode"])
    assert not any("network" in w for w in c.warnings)


def test_dirty_harness_warning(tmp_path):
    runs = two_harness_runs()
    runs[0][2]["harnessDirty"] = True
    _, records = build(tmp_path, runs)
    c = build_comparison(records, ["Pi", "OpenCode"])
    assert any("1 run(s) were produced with a dirty harness" in w for w in c.warnings)


def test_mixed_models_and_unevaluated_warnings(tmp_path):
    runs = two_harness_runs()
    runs[1][2]["model"] = "other-model"
    runs[2][2]["outcome"] = None
    _, records = build(tmp_path, runs)
    c = build_comparison(records, ["Pi", "OpenCode"])
    assert any("mixed models" in w for w in c.warnings)
    assert any("no valid evaluation" in w for w in c.warnings)
    assert c.summaries[1].evaluated == 1


def test_multiple_attempts_per_task_are_flagged(tmp_path):
    runs = two_harness_runs() + [("pi-a2", "pi-swebench-reference", {"instanceId": "p__a-1"})]
    _, records = build(tmp_path, runs)
    c = build_comparison(records, ["Pi", "OpenCode"])
    assert c.summaries[0].runs == 3 and c.summaries[0].tasks == 2
    assert any("multiple attempts" in w for w in c.warnings)


def test_errors_for_unknown_or_single_harness(tmp_path):
    _, records = build(tmp_path, two_harness_runs())
    with pytest.raises(NoComparableRuns, match="Nope"):
        build_comparison(records, ["Pi", "Nope"])
    with pytest.raises(NoComparableRuns, match="two distinct"):
        build_comparison(records, ["Pi", "pi"])


def test_no_common_tasks(tmp_path):
    runs = [
        ("pi-a", "pi-swebench-reference", {"instanceId": "p__a-1"}),
        ("oc-b", "opencode-swebench", {"instanceId": "p__b-2"}),
    ]
    _, records = build(tmp_path, runs)
    with pytest.raises(NoComparableRuns, match="no tasks in common"):
        build_comparison(records, ["Pi", "OpenCode"])


def test_cli_compare_output_percentage_points_and_na(tmp_path):
    raw, _ = build(tmp_path, two_harness_runs())
    result = runner.invoke(
        app, ["compare", "--raw-dir", str(raw), "--harness", "Pi", "--harness", "OpenCode"],
        env=WIDE,
    )  # fmt: skip
    assert result.exit_code == 0
    assert "100.0%" in result.output and "50.0%" in result.output
    assert "-50.0 percentage points" in result.output
    line = next(ln for ln in result.output.splitlines() if "Tool calls" in ln)
    assert line.count("N/A") == 3  # unavailable in historical data: N/A, not 0
    assert "warning: mixed network policies" in result.output
    assert "no statistical significance" in result.output


def test_cli_compare_relative_difference_when_metrics_available(tmp_path):
    raw, _ = build(tmp_path, two_harness_runs())
    # Pi 300 s vs OpenCode 300 s -> +0.0% is a real value here
    result = runner.invoke(
        app, ["compare", "--raw-dir", str(raw), "--harness", "Pi", "--harness", "OpenCode"],
        env=WIDE,
    )  # fmt: skip
    line = next(ln for ln in result.output.splitlines() if "Total duration" in ln)
    assert "300" in line and "+0.0%" in line


def test_cli_compare_requires_two_harnesses(tmp_path):
    raw, _ = build(tmp_path, two_harness_runs())
    result = runner.invoke(app, ["compare", "--raw-dir", str(raw), "--harness", "Pi"], env=WIDE)
    assert result.exit_code == 2
    assert "at least two harnesses" in result.output


def test_cli_compare_missing_results_dir(tmp_path):
    result = runner.invoke(
        app, ["compare", "--raw-dir", str(tmp_path / "x"), "--harness", "Pi", "--harness", "OC"],
        env=WIDE,
    )  # fmt: skip
    assert result.exit_code == 1
    assert "No results directory found" in result.output
