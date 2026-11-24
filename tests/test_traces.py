import json

import pytest

from agent_harness_lab.traces import (
    ModelTurnEvent,
    ToolCallEvent,
    ToolResultEvent,
    TraceReport,
    UnsupportedHarness,
    read_events,
)
from tests.trace_fixtures import (
    PI_SYSTEM_END,
    PI_USER_END,
    jl,
    oc_finish,
    oc_tool,
    pi_end,
    pi_start,
    pi_turn,
)


def events(tmp_path, text, harness):
    path = tmp_path / "trace.jsonl"
    path.write_text(text, encoding="utf-8")
    report = TraceReport(path=path)
    return list(read_events(path, harness, report)), report


# ---- Pi --------------------------------------------------------------------
def test_pi_tool_call_event(tmp_path):
    evs, _ = events(tmp_path, jl(pi_start("bash", command="ls")), "Pi")
    assert evs == [ToolCallEvent(name="bash", command="ls")]


def test_pi_command_only_exposed_for_bash(tmp_path):
    evs, _ = events(tmp_path, jl(pi_start("read", command="pytest")), "Pi")
    assert evs == [ToolCallEvent(name="read", command=None)]


def test_pi_model_usage_tokens_and_cost(tmp_path):
    turn = pi_turn(inp=3, out=45, reasoning=7, cache_read=11, cost=0.5)
    evs, _ = events(tmp_path, jl(turn), "Pi")
    assert evs == [
        ModelTurnEvent(
            input_tokens=3,
            output_tokens=45,
            reasoning_tokens=7,
            cache_read_tokens=11,
            cache_write_tokens=1351,
            cost_usd=0.5,
        )
    ]


def test_pi_only_assistant_messages_with_a_model_are_turns(tmp_path):
    no_model = pi_turn()
    del no_model["message"]["model"]
    evs, _ = events(tmp_path, jl(PI_SYSTEM_END, PI_USER_END, no_model), "Pi")
    assert evs == []


@pytest.mark.parametrize(
    "end",
    [
        {"type": "tool_execution_end", "isError": True},
        {"type": "tool_execution_end", "error": "boom"},
        {"type": "tool_execution_end", "result": {"isError": True}},
    ],
)
def test_pi_tool_error_variants(tmp_path, end):
    evs, _ = events(tmp_path, jl(end), "Pi")
    assert evs == [ToolResultEvent(is_error=True)]


def test_pi_successful_tool_result_is_not_error(tmp_path):
    evs, _ = events(tmp_path, jl(pi_end(is_error=False)), "Pi")
    assert evs == [ToolResultEvent(is_error=False)]


def test_pi_missing_optional_usage_fields_are_zero(tmp_path):
    turn = {"type": "message_end", "message": {"role": "assistant", "model": "m"}}
    evs, _ = events(tmp_path, jl(turn), "Pi")
    assert evs == [ModelTurnEvent()]


# ---- OpenCode ----------------------------------------------------------------
def test_opencode_tool_call_and_result(tmp_path):
    evs, _ = events(tmp_path, jl(oc_tool("shell", command="npm test")), "OpenCode")
    assert evs == [
        ToolCallEvent(name="shell", command="npm test"),
        ToolResultEvent(is_error=False),
    ]


def test_opencode_tool_error_status(tmp_path):
    evs, _ = events(tmp_path, jl(oc_tool("grep", status="error", pattern="x")), "OpenCode")
    assert evs[1] == ToolResultEvent(is_error=True)
    assert evs[0].command is None


def test_opencode_command_only_for_shell_tools(tmp_path):
    evs, _ = events(tmp_path, jl(oc_tool("read", command="pytest")), "OpenCode")
    assert evs[0].command is None


def test_opencode_step_finish_usage(tmp_path):
    finish = oc_finish(inp=3, out=23, reasoning=2, read=9, write=4565, cost=0.25)
    evs, _ = events(tmp_path, jl(finish), "OpenCode")
    assert evs == [
        ModelTurnEvent(
            input_tokens=3,
            output_tokens=23,
            reasoning_tokens=2,
            cache_read_tokens=9,
            cache_write_tokens=4565,
            cost_usd=0.25,
        )
    ]


def test_opencode_step_finish_without_tokens_is_not_a_turn(tmp_path):
    evs, _ = events(tmp_path, jl({"type": "step_finish", "part": {"cost": 1}}), "OpenCode")
    assert evs == []


def test_opencode_optional_token_fields(tmp_path):
    finish = {"type": "step_finish", "part": {"tokens": {"input": 5}}}  # no cache/cost/output
    evs, _ = events(tmp_path, jl(finish), "OpenCode")
    assert evs == [ModelTurnEvent(input_tokens=5)]


def test_unknown_event_types_are_ignored(tmp_path):
    evs, _ = events(tmp_path, jl({"type": "step_start"}, {"type": "text"}), "OpenCode")
    assert evs == []


# ---- Reader robustness -------------------------------------------------------
def test_malformed_line_is_skipped_with_warning(tmp_path):
    text = jl(pi_start("bash", command="ls")) + "{not json\n" + jl(pi_start("read"))
    evs, report = events(tmp_path, text, "Pi")
    assert len(evs) == 2
    assert report.malformed_lines == 1
    assert report.warnings[0].kind == "malformed_line" and report.warnings[0].line == 2
    assert "1 malformed line(s) skipped, first at line 2" in report.summary()
    assert report.usable


def test_truncated_final_line_is_tolerated(tmp_path):
    full = jl(pi_start("bash", command="ls"))
    evs, report = events(tmp_path, full + '{"type": "tool_execution_st', "Pi")
    assert len(evs) == 1
    assert report.truncated_final_line and report.malformed_lines == 0
    assert "truncated final line ignored" in report.summary()


def test_valid_final_line_without_newline_is_accepted(tmp_path):
    text = json.dumps(pi_start("bash", command="ls"))  # no trailing newline
    evs, report = events(tmp_path, text, "Pi")
    assert len(evs) == 1 and report.summary() is None


def test_non_object_lines_are_skipped(tmp_path):
    evs, report = events(tmp_path, "42\n[1]\n" + jl(pi_start("read")), "Pi")
    assert len(evs) == 1 and report.warning_count == 2
    assert "non-object" in report.summary()


def test_empty_trace_is_not_usable(tmp_path):
    evs, report = events(tmp_path, "", "Pi")
    assert evs == [] and not report.usable
    assert report.summary() == "empty trace"


def test_trace_with_only_garbage_is_not_usable(tmp_path):
    _, report = events(tmp_path, "garbage\n", "Pi")
    assert not report.usable and "no valid JSON events" in report.summary()


def test_trace_with_events_but_no_metric_events_is_usable(tmp_path):
    _, report = events(tmp_path, jl({"type": "agent_start"}), "Pi")
    assert report.usable  # valid trace; genuine zeros are possible


def test_missing_trace_reports_missing(tmp_path):
    report = TraceReport(path=tmp_path / "trace.jsonl")
    assert list(read_events(report.path, "Pi", report)) == []
    assert not report.exists and not report.usable
    assert "missing" in report.summary()


def test_warning_list_is_capped_but_counted(tmp_path):
    _, report = events(tmp_path, "bad\n" * 100, "Pi")
    assert report.malformed_lines == 100 and len(report.warnings) == 20


def test_reader_is_lazy_and_never_modifies_the_file(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text(jl(pi_start("read")) * 3)
    before = path.read_bytes()
    report = TraceReport(path=path)
    stream = read_events(path, "Pi", report)
    assert report.lines == 0  # nothing read until iterated
    next(stream)
    assert report.lines == 1  # one line at a time, not the whole file
    list(stream)
    assert path.read_bytes() == before


def test_unsupported_harness(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text("")
    with pytest.raises(UnsupportedHarness):
        list(read_events(path, "Mystery", TraceReport(path=path)))
