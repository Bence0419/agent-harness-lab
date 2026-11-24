import json

import pytest

from agent_harness_lab.harnesses import (
    PROMPT_PLACEHOLDER,
    HarnessError,
    available_harnesses,
    get_harness,
)
from agent_harness_lab.harnesses.discovery import discover_executable, pinned_version
from agent_harness_lab.harnesses.opencode import render_config
from agent_harness_lab.harnesses.pi import render_models_json
from agent_harness_lab.providers import load_providers_file, resolve_model

PI_FIXED = [
    "--print", "--mode", "json", "--no-session", "--no-extensions", "--no-skills",
    "--no-prompt-templates", "--no-context-files", "--no-approve",
    "--tools", "read,bash,edit,write",
]  # fmt: skip


def model(ref, reasoning=None, providers=None, env=None):
    return resolve_model(ref, reasoning, providers or {}, env or {})


@pytest.fixture
def lab(providers_file):
    return load_providers_file(providers_file)


# ---- registry ------------------------------------------------------------------
def test_registry():
    assert available_harnesses() == ["opencode", "pi"]
    assert get_harness("pi").display_name == "Pi"
    assert get_harness("OpenCode").display_name == "OpenCode"  # case-insensitive


def test_unknown_harness():
    with pytest.raises(HarnessError, match="unknown harness 'aider'.*opencode, pi"):
        get_harness("aider")


# ---- Pi ------------------------------------------------------------------------
def test_pi_invocation_argv():
    inv = get_harness("pi").build_invocation(model("openai/gpt-6-astra", "medium"))
    assert inv.argv == [
        "/opt/pi-node_modules/.bin/pi", "--provider", "openai", "--model", "gpt-6-astra",
        "--thinking", "medium", *PI_FIXED, PROMPT_PLACEHOLDER,
    ]  # fmt: skip
    assert inv.executable == "/opt/pi-node_modules/.bin/pi"
    assert inv.secret_env == ["OPENAI_API_KEY"]
    assert inv.config_files == []
    assert inv.env["PATH"].startswith("/opt/node24/bin:")
    assert inv.output.trace_format == "Pi" and inv.output.stderr_file == "pi.stderr.log"
    assert inv.output.stdout_file == "trace.jsonl"


def test_pi_without_reasoning_omits_thinking():
    argv = get_harness("pi").build_invocation(model("openai/gpt-6-astra")).argv
    assert "--thinking" not in argv


def test_pi_never_enables_extra_features():
    argv = get_harness("pi").build_invocation(model("openai/m", "high")).argv
    for flag in ("--extension", "--skill", "--approve", "--session", "-e", "-a"):
        assert flag not in argv


def test_pi_reasoning_validation_uses_pis_own_levels():
    pi = get_harness("pi")
    for ok in ("off", "minimal", "low", "medium", "high", "xhigh", "max", None):
        pi.validate_reasoning(ok)
    with pytest.raises(HarnessError, match="does not support reasoning level 'ultra'"):
        pi.build_invocation(model("openai/m", "ultra"))


def test_pi_prompt_is_a_single_final_argv_element():
    argv = get_harness("pi").build_invocation(model("openai/m", "low")).argv
    assert argv[-1] == PROMPT_PLACEHOLDER and argv.count(PROMPT_PLACEHOLDER) == 1
    assert all(isinstance(a, str) for a in argv)  # argv list, never a shell string


def test_pi_compatible_provider_renders_models_json(lab):
    m = model("example-lab/Org/Model-7B-FP8", None, lab)
    inv = get_harness("pi").build_invocation(m)
    assert inv.argv[1:5] == ["--provider", "example-lab", "--model", "Org/Model-7B-FP8"]
    assert inv.secret_env == ["EXAMPLE_LAB_API_KEY"]
    (f,) = inv.config_files
    assert f.container_path == "/root/.pi/agent/models.json"
    doc = json.loads(f.content)
    p = doc["providers"]["example-lab"]
    assert p["baseUrl"] == "https://llm.example.org/v1"
    assert p["apiKey"] == "$EXAMPLE_LAB_API_KEY"  # a variable reference, not a value
    assert (p["api"], p["authHeader"]) == ("openai-completions", True)
    assert p["compat"] == {"supportsReasoningEffort": False, "supportsDeveloperRole": False}
    (entry,) = p["models"]
    assert entry["id"] == "Org/Model-7B-FP8" and entry["name"] == "Example 7B"
    assert (entry["contextWindow"], entry["maxTokens"]) == (32768, 4096)
    assert entry["input"] == ["text"]
    assert "reasoning" not in entry  # capability unknown: not asserted either way
    assert entry["cost"] == {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}


def test_pi_compatible_provider_needs_model_limits(lab):
    with pytest.raises(HarnessError, match="context_window and max_output_tokens"):
        render_models_json(
            model("env-lab/m", None, lab, {"ENV_LAB_BASE_URL": "https://h.example/v1"})
        )


# ---- OpenCode ------------------------------------------------------------------
def test_opencode_invocation_argv():
    inv = get_harness("opencode").build_invocation(model("openai/gpt-6-astra", "medium"))
    assert inv.argv == [
        "/usr/local/bin/opencode", "run", "--standalone", "--auto", "--agent", "build",
        "--model", "openai/gpt-6-astra#medium", "--format", "json", PROMPT_PLACEHOLDER,
    ]  # fmt: skip
    assert inv.env == {"OPENCODE_CONFIG": "/etc/opencode/opencode.json"}
    assert inv.secret_env == ["OPENAI_API_KEY"]
    assert inv.output.trace_format == "OpenCode"
    assert inv.output.stderr_file == "opencode.stderr.log"


def test_opencode_reasoning_becomes_variant_syntax_only_in_the_adapter():
    oc = get_harness("opencode")
    assert "openai/m#high" in oc.build_invocation(model("openai/m", "high")).argv
    assert "openai/m" in oc.build_invocation(model("openai/m")).argv  # no '#' without reasoning
    m = model("openai/m", "high")
    assert "#" not in m.model.model and m.model.reasoning == "high"  # generic spec stays clean


def test_opencode_openai_config_is_policy_only():
    (f,) = get_harness("opencode").build_invocation(model("openai/m")).config_files
    assert f.container_path == "/etc/opencode/opencode.json"
    doc = json.loads(f.content)
    assert set(doc) == {"$schema", "permissions"}
    assert {(p["action"], p["effect"]) for p in doc["permissions"]} == {
        ("webfetch", "deny"),
        ("websearch", "deny"),
    }


def test_opencode_compatible_provider_renders_provider_block(lab):
    m = model("example-lab/Org/Model-7B-FP8", None, lab)
    inv = get_harness("opencode").build_invocation(m)
    assert "example-lab/Org/Model-7B-FP8" in inv.argv  # nested model id preserved
    doc = json.loads(render_config(m))
    p = doc["provider"]["example-lab"]
    assert p["npm"] == "@ai-sdk/openai-compatible" and p["name"] == "Example Lab"
    assert p["options"] == {
        "baseURL": "https://llm.example.org/v1",
        "apiKey": "{env:EXAMPLE_LAB_API_KEY}",
    }
    assert p["models"]["Org/Model-7B-FP8"] == {
        "name": "Example 7B",
        "limit": {"context": 32768, "output": 4096},
    }
    assert "permissions" in doc


def test_rendered_configs_never_contain_secret_values(lab, monkeypatch):
    monkeypatch.setenv("EXAMPLE_LAB_API_KEY", "sk-should-never-appear")
    m = model("example-lab/Org/Model-7B-FP8", None, lab)
    for name in ("pi", "opencode"):
        inv = get_harness(name).build_invocation(m)
        assert "sk-should-never-appear" not in inv.model_dump_json()


# ---- discovery reuse -------------------------------------------------------------
def test_discovery_prefers_path_then_project_local(healthy_probe):
    healthy_probe.root = "/fake/project"
    local = "/fake/project/node_modules/.bin/pi"
    healthy_probe.executable_files.add(local)
    found = discover_executable(healthy_probe, "pi")
    assert (found.path, found.source) == ("/fake/pi", "PATH")  # global wins
    del healthy_probe.executables["pi"]
    found = discover_executable(healthy_probe, "pi")
    assert (found.path, found.source) == (local, "project-local")
    healthy_probe.executable_files.clear()
    assert discover_executable(healthy_probe, "pi") is None


def test_pinned_version_reads_package_json(plan_probe):
    assert pinned_version(plan_probe, "@earendil-works/pi-coding-agent") == "1.0.2"
    assert pinned_version(plan_probe, "@opencode/cli") == "2.2.0"
    assert pinned_version(plan_probe, "missing") is None


def test_doctor_and_run_share_one_discovery_implementation():
    from agent_harness_lab.doctor import checks

    assert checks.discover_executable is discover_executable


def test_pi_models_json_renders_explicit_capability_only(lab):
    for ref, expected in (("example-lab/Org/Reasoner", True), ("example-lab/Org/Plain", False)):
        doc = json.loads(render_models_json(model(ref, None, lab)))
        (entry,) = doc["providers"]["example-lab"]["models"]
        assert entry["reasoning"] is expected
