import csv

from agent_harness_lab.artifacts.legacy import load_run
from agent_harness_lab.results.trace_metrics import AstraCsvMetrics
from tests.factories import make_run

FIELDS = [
    "task", "harness", "resolved", "duration_s", "model_turns", "tool_calls", "tool_errors",
    "verification_calls", "input_tokens", "output_tokens", "reasoning_tokens",
    "cache_read_tokens", "cache_write_tokens", "cost_usd", "files_changed", "insertions",
    "deletions",
]  # fmt: skip


def write_csv(path, **row):
    base = dict.fromkeys(FIELDS, "0")
    base.update(task="proj__repo-1", harness="Pi", resolved="True", duration_s="150")
    base.update(model_turns="7", tool_calls="9", cost_usd="0.5", **row)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerow(base)


def test_csv_metrics_attach_to_matching_astra_run(tmp_path):
    rec = load_run(make_run(tmp_path, "r1"))
    write_csv(tmp_path / "m.csv")
    out, warnings = AstraCsvMetrics(tmp_path / "m.csv").enrich([rec])
    assert warnings == []
    assert (out[0].model_turns, out[0].tool_calls, out[0].cost_usd) == (7, 9, 0.5)


def test_csv_does_not_touch_uncovered_runs(tmp_path):
    # Same task/harness as the CSV row, but a different model/reference.
    rec = load_run(make_run(tmp_path, "r1", "pi-swebench-baseline"))
    write_csv(tmp_path / "m.csv")
    out, _ = AstraCsvMetrics(tmp_path / "m.csv").enrich([rec])
    assert out[0].model_turns is None and out[0].cost_usd is None


def test_csv_disagreeing_with_run_is_not_attached(tmp_path):
    rec = load_run(make_run(tmp_path, "r1", outcome="unresolved"))
    write_csv(tmp_path / "m.csv")  # CSV claims resolved=True
    out, warnings = AstraCsvMetrics(tmp_path / "m.csv").enrich([rec])
    assert out[0].tool_calls is None
    assert "disagrees" in warnings[0]


def test_csv_ambiguous_key_is_not_attached(tmp_path):
    a = load_run(make_run(tmp_path, "r1"))
    b = load_run(make_run(tmp_path, "r2"))
    write_csv(tmp_path / "m.csv")
    out, warnings = AstraCsvMetrics(tmp_path / "m.csv").enrich([a, b])
    assert all(r.tool_calls is None for r in out)
    assert "share the CSV key" in warnings[0]


def test_missing_csv_is_silent(tmp_path):
    rec = load_run(make_run(tmp_path, "r1"))
    out, warnings = AstraCsvMetrics(tmp_path / "absent.csv").enrich([rec])
    assert out == [rec] and warnings == []
