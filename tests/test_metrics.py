import subprocess
from pathlib import Path

import pytest

from agent_harness_lab.metrics import METRICS_SCHEMA_V1, extract_trajectory
from agent_harness_lab.metrics.patch import (
    GitApplyNumstat,
    PatchStatError,
    parse_numstat,
    patch_stats,
)
from agent_harness_lab.metrics.tools import classify_tool
from agent_harness_lab.metrics.verification import is_verification_command
from agent_harness_lab.traces import ModelTurnEvent, ToolCallEvent, ToolResultEvent


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("read", "read"),
        ("bash", "bash"),
        ("shell", "bash"),
        ("edit", "edit"),
        ("patch", "edit"),
        ("apply_patch", "edit"),
        ("write", "write"),
        # everything else is "other", including search tools (documented v1 behaviour)
        ("grep", "other"),
        ("glob", "other"),
        ("execute", "other"),
        ("list", "other"),
        ("Read", "other"),  # classification is case-sensitive
        ("unknown", "other"),
        (None, "other"),
    ],
)
def test_tool_classification_matches_v1(name, expected):
    assert classify_tool(name) == expected


@pytest.mark.parametrize(
    "command",
    [
        "pytest -q tests/",
        "python -m pytest",
        "npx jest src/foo.test.js",
        "vitest run",
        "mocha test/",
        "karma start",
        "cargo test",
        "go test ./...",
        "npm test",
        "npm run test",
        "pnpm test",
        "yarn test",
        "bun test",
        "yarn run test --watch=false",
        "NPM TEST",  # case-insensitive
        "cd x && pytest",
        "grep pytest setup.cfg",  # known v1 imperfection: substring match
    ],
)
def test_verification_commands_detected(command):
    assert is_verification_command(command)


@pytest.mark.parametrize(
    "command",
    [
        "ls -la",
        "git diff",
        "node build.js",
        "npm install",
        "npm run build",
        "cargo build",
        "go build ./...",
        # known v1 imperfections: real test runners the heuristic does not recognise
        "mvn test",
        "./gradlew test",
        "make test",
        "tox",
        "",
        None,
    ],
)
def test_non_verification_commands_not_detected(command):
    assert not is_verification_command(command)


def test_extract_counts_tools_by_class():
    events = [
        ToolCallEvent("read"),
        ToolCallEvent("bash", "ls"),
        ToolCallEvent("shell", "pytest"),
        ToolCallEvent("edit"),
        ToolCallEvent("apply_patch"),
        ToolCallEvent("write"),
        ToolCallEvent("grep"),
    ]
    m = extract_trajectory(events)
    assert (m.tool_calls, m.read_calls, m.bash_calls, m.edit_calls) == (7, 1, 2, 2)
    assert (m.write_calls, m.other_tool_calls, m.verification_calls) == (1, 1, 1)


def test_extract_counts_only_error_results():
    m = extract_trajectory([ToolResultEvent(True), ToolResultEvent(False), ToolResultEvent(True)])
    assert m.tool_errors == 2 and m.tool_calls == 0


def test_extract_aggregates_tokens_and_cost():
    turns = [
        ModelTurnEvent(1, 10, 2, 100, 5, 0.1),
        ModelTurnEvent(2, 20, 3, 200, 6, 0.2),
        ModelTurnEvent(),
    ]
    m = extract_trajectory(turns)
    assert m.model_turns == 3
    assert (m.input_tokens, m.output_tokens, m.reasoning_tokens) == (3, 30, 5)
    assert (m.cache_read_tokens, m.cache_write_tokens) == (300, 11)
    assert m.cost_usd == pytest.approx(0.3)


def test_cost_is_rounded_to_six_decimals_like_v1():
    m = extract_trajectory([ModelTurnEvent(cost_usd=0.1234564), ModelTurnEvent(cost_usd=0.0000004)])
    assert m.cost_usd == 0.123457


def test_metrics_schema_identifier():
    assert METRICS_SCHEMA_V1 == "ahl-metrics-v1"
    assert extract_trajectory([]).metrics_schema == "ahl-metrics-v1"


def test_empty_event_stream_yields_genuine_zeros():
    m = extract_trajectory([])
    assert m.tool_calls == 0 and m.model_turns == 0 and m.cost_usd == 0.0


# ---- patch stats -------------------------------------------------------------
NUMSTAT = (
    "10\t2\tsrc/main.js\n"
    "30\t0\ttest/main.test.js\n"
    "-\t-\tassets/logo.png\n"
    "4\t1\tpkg/__tests__/a.js\n"
    "2\t2\tlib/util.spec.ts\n"
    "1\t1\tcontest/readme.md\n"
)


def test_parse_numstat():
    s = parse_numstat(NUMSTAT)
    assert s.files_changed == 6
    assert s.test_files_changed == 3  # test/, __tests__/, .spec.; "contest/" is not a test dir
    assert s.non_test_files_changed == 3
    assert (s.insertions, s.deletions) == (10 + 30 + 4 + 2 + 1, 2 + 0 + 1 + 2 + 1)  # binary adds 0


def test_parse_numstat_rejects_garbage():
    with pytest.raises(PatchStatError):
        parse_numstat("not a numstat line\n")


class FakeNumstat:
    def __init__(self, text="", error=False):
        self.text, self.error, self.calls = text, error, []

    def numstat(self, patch):
        self.calls.append(patch)
        if self.error:
            raise PatchStatError("boom")
        return self.text


def test_patch_stats_via_fake_probe(tmp_path):
    patch = tmp_path / "model.patch"
    patch.write_text("diff --git a/x b/x\n")
    probe = FakeNumstat("3\t1\tsrc/a.js\n")
    s = patch_stats(patch, probe)
    assert (s.files_changed, s.insertions, s.deletions) == (1, 3, 1)
    assert probe.calls == [patch]


def test_missing_patch_is_unavailable_not_zero(tmp_path):
    probe = FakeNumstat()
    assert patch_stats(tmp_path / "model.patch", probe) is None
    assert probe.calls == []


def test_empty_patch_is_genuine_zero_and_skips_git(tmp_path):
    patch = tmp_path / "model.patch"
    patch.write_text("")
    probe = FakeNumstat()
    s = patch_stats(patch, probe)
    assert s.files_changed == 0 and s.insertions == 0
    assert probe.calls == []


def test_probe_failure_propagates(tmp_path):
    patch = tmp_path / "model.patch"
    patch.write_text("x")
    with pytest.raises(PatchStatError):
        patch_stats(patch, FakeNumstat(error=True))


def test_git_probe_uses_absolute_patch_path_and_neutral_cwd(monkeypatch, tmp_path):
    """Pins the two compatibility details; git itself is faked."""
    seen = {}

    def fake_run(argv, **kw):
        seen["argv"], seen["cwd"] = argv, kw["cwd"]
        return subprocess.CompletedProcess(argv, 0, stdout="1\t0\tf\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)
    out = GitApplyNumstat().numstat(Path("rel/model.patch"))
    assert out == "1\t0\tf\n"
    assert seen["argv"][:3] == ["git", "apply", "--numstat"]
    assert Path(seen["argv"][3]).is_absolute()
    assert Path(seen["cwd"]) != tmp_path  # not the repo/run directory


def test_git_probe_wraps_failures(monkeypatch, tmp_path):
    def boom(argv, **kw):
        raise subprocess.CalledProcessError(128, argv)

    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(PatchStatError):
        GitApplyNumstat().numstat(tmp_path / "p.patch")
