"""Opt-in regression: native extraction must reproduce the frozen Astra CSV, per run.

Skipped unless AHL_REAL_DATA_REGRESSION=1 (it reads the real results/raw tree and
runs real `git apply --numstat`, so it is not a unit test).
"""

import csv
import os

import pytest

from agent_harness_lab.paths import default_raw_dir
from agent_harness_lab.results.native_metrics import NativeTraceMetrics
from agent_harness_lab.results.service import attach_metrics, filter_runs, load_runs

pytestmark = pytest.mark.skipif(
    os.environ.get("AHL_REAL_DATA_REGRESSION") != "1",
    reason="set AHL_REAL_DATA_REGRESSION=1 to compare against the real frozen CSV",
)

INT_FIELDS = (
    "model_turns", "tool_calls", "tool_errors", "read_calls", "bash_calls", "edit_calls",
    "write_calls", "other_tool_calls", "verification_calls", "input_tokens", "output_tokens",
    "reasoning_tokens", "cache_read_tokens", "cache_write_tokens", "files_changed",
    "test_files_changed", "non_test_files_changed", "insertions", "deletions",
)  # fmt: skip


def test_native_matches_frozen_csv_per_run():
    raw = default_raw_dir()
    csv_path = raw.parent / "analysis" / "astra-trajectory-metrics.csv"
    if not (raw.is_dir() and csv_path.is_file()):
        pytest.skip("real results not present")
    rows = list(csv.DictReader(csv_path.open(newline="")))
    records = filter_runs(load_runs(raw).records, models=["gpt-6-astra"])
    records, warnings = attach_metrics(records, NativeTraceMetrics())
    assert warnings == []
    by_key = {(r.task, r.harness): r for r in records}
    assert len(rows) == len(by_key) == 22
    for row in rows:
        r = by_key[(row["task"], row["harness"])]
        where = f"{row['task']}/{row['harness']}"
        assert str(r.resolved) == row["resolved"], where
        assert r.duration_seconds == int(row["duration_s"]), where
        for f in INT_FIELDS:
            assert getattr(r, f) == int(row[f]), f"{where}: {f}"
        assert r.cost_usd == pytest.approx(float(row["cost_usd"]), abs=5e-7), where
