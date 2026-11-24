#!/usr/bin/env python3

import csv
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/raw/swebench"
OUTPUT = ROOT / "results/analysis/astra-trajectory-metrics.csv"

TEST_RE = re.compile(
    r"(?:pytest|jest|vitest|mocha|karma|cargo\s+test|go\s+test|"
    r"(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?test)",
    re.I,
)

TEST_PATH_RE = re.compile(
    r"(^|/)(?:test|tests|__tests__)(/|$)|\.(?:test|spec)\.",
    re.I,
)


def iso_duration_seconds(start, end):
    if not start or not end:
        return None
    a = datetime.fromisoformat(start.replace("Z", "+00:00"))
    b = datetime.fromisoformat(end.replace("Z", "+00:00"))
    return int((b - a).total_seconds())


def patch_stats(path):
    if not path.exists() or path.stat().st_size == 0:
        return 0, 0, 0, 0, 0

    proc = subprocess.run(
        ["git", "apply", "--numstat", str(path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    files = []
    ins = 0
    dels = 0

    for line in proc.stdout.splitlines():
        added, removed, filename = line.split("\t", 2)
        files.append(filename)

        if added.isdigit():
            ins += int(added)
        if removed.isdigit():
            dels += int(removed)

    test_files = sum(bool(TEST_PATH_RE.search(f)) for f in files)

    return len(files), test_files, len(files) - test_files, ins, dels


def add_usage(total, usage):
    total["input_tokens"] += usage.get("input", 0) or 0
    total["output_tokens"] += usage.get("output", 0) or 0
    total["reasoning_tokens"] += usage.get("reasoning", 0) or 0
    total["cache_read_tokens"] += usage.get("cacheRead", 0) or 0
    total["cache_write_tokens"] += usage.get("cacheWrite", 0) or 0

    cost = usage.get("cost") or {}
    total["cost_usd"] += cost.get("total", 0) or 0


def extract_pi(events):
    m = empty_metrics()

    for e in events:
        if e.get("type") == "message_end":
            msg = e.get("message") or {}
            if msg.get("role") == "assistant" and msg.get("model"):
                m["model_turns"] += 1
                add_usage(m, msg.get("usage") or {})

        elif e.get("type") == "tool_execution_start":
            m["tool_calls"] += 1
            name = e.get("toolName", "unknown")
            count_tool(m, name)

            if name == "bash":
                command = (e.get("args") or {}).get("command", "")
                if TEST_RE.search(command):
                    m["verification_calls"] += 1

        elif e.get("type") == "tool_execution_end":
            if (
                e.get("isError") is True
                or e.get("error") is not None
                or (e.get("result") or {}).get("isError") is True
            ):
                m["tool_errors"] += 1

    return m


def extract_opencode(events):
    m = empty_metrics()

    for e in events:
        if e.get("type") == "step_finish":
            part = e.get("part") or {}
            tokens = part.get("tokens")

            if tokens:
                m["model_turns"] += 1

                usage = {
                    "input": tokens.get("input", 0),
                    "output": tokens.get("output", 0),
                    "reasoning": tokens.get("reasoning", 0),
                    "cacheRead": (tokens.get("cache") or {}).get("read", 0),
                    "cacheWrite": (tokens.get("cache") or {}).get("write", 0),
                    "cost": {"total": part.get("cost", 0)},
                }
                add_usage(m, usage)

        elif e.get("type") == "tool_use":
            m["tool_calls"] += 1

            part = e.get("part") or {}
            name = part.get("tool", "unknown")
            count_tool(m, name)

            state = part.get("state") or {}

            if state.get("status") == "error":
                m["tool_errors"] += 1

            if name in {"shell", "bash"}:
                command = (state.get("input") or {}).get("command", "")
                if TEST_RE.search(command):
                    m["verification_calls"] += 1

    return m


def empty_metrics():
    return {
        "model_turns": 0,
        "tool_calls": 0,
        "tool_errors": 0,
        "read_calls": 0,
        "bash_calls": 0,
        "edit_calls": 0,
        "write_calls": 0,
        "other_tool_calls": 0,
        "verification_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "cost_usd": 0.0,
    }


def count_tool(m, name):
    if name == "read":
        m["read_calls"] += 1
    elif name in {"bash", "shell"}:
        m["bash_calls"] += 1
    elif name in {"edit", "patch", "apply_patch"}:
        m["edit_calls"] += 1
    elif name == "write":
        m["write_calls"] += 1
    else:
        m["other_tool_calls"] += 1


def load_jsonl(path):
    events = []
    with path.open() as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    return events


rows = []

for meta_path in RESULTS.glob("*/*/metadata.json"):
    meta = json.loads(meta_path.read_text())

    if meta.get("model") != "gpt-6-astra":
        continue

    reference = meta.get("reference", "")
    if "frontier-astra-v1" not in reference:
        continue

    run_dir = meta_path.parent
    trace_path = run_dir / "trace.jsonl"
    evaluation_path = run_dir / "evaluation.json"

    events = load_jsonl(trace_path)

    harness = "OpenCode" if reference.startswith("opencode") else "Pi"

    metrics = (
        extract_opencode(events)
        if harness == "OpenCode"
        else extract_pi(events)
    )

    resolved = None
    if evaluation_path.exists():
        evaluation = json.loads(evaluation_path.read_text())
        resolved = evaluation.get("resolved")

    files, test_files, non_test_files, ins, dels = patch_stats(
        run_dir / "model.patch"
    )

    row = {
        "task": meta["instanceId"],
        "harness": harness,
        "resolved": resolved,
        "duration_s": iso_duration_seconds(
            meta.get("startedAt"), meta.get("finishedAt")
        ),
        **metrics,
        "files_changed": files,
        "test_files_changed": test_files,
        "non_test_files_changed": non_test_files,
        "insertions": ins,
        "deletions": dels,
        "trace_kb": round(trace_path.stat().st_size / 1024, 1),
    }

    row["cost_usd"] = round(row["cost_usd"], 6)
    rows.append(row)

rows.sort(key=lambda r: (r["task"], r["harness"]))

if len(rows) != 22:
    raise SystemExit(
        f"Expected 22 Astra runs (11 tasks x 2 harnesses), found {len(rows)}"
    )

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUTPUT}")
