# ruff: noqa: E501
"""Command parity: the adapters must express the frozen runners' treatments.

Instead of hand-copied goldens, the expected values are PARSED from the frozen runner scripts
and config files in this repository, so any drift in either direction fails. Nothing is executed.

Historical differences that the normalized plan does not (or need not) represent are asserted
explicitly in ``HISTORICAL`` rather than being normalised away.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass

import pytest

from agent_harness_lab.benchmarks.swebench_multilingual import SWEBenchMultilingual
from agent_harness_lab.harnesses import PROMPT_PLACEHOLDER, get_harness
from agent_harness_lab.paths import find_project_root
from agent_harness_lab.providers import ModelInfo, ProviderSpec, resolve_model
from agent_harness_lab.runplan.builder import build_run_plan
from agent_harness_lab.runplan.models import NetworkMode
from agent_harness_lab.runplan.prompt import PREAMBLE

ROOT = find_project_root()
pytestmark = pytest.mark.skipif(ROOT is None, reason="needs a source checkout with scripts/")

QWEN_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"


@dataclass(frozen=True)
class Combo:
    id: str
    script: str
    harness: str
    provider: str
    model: str
    reasoning: str | None
    network: NetworkMode  # what the frozen runner actually did
    has_timeout: bool  # whether the frozen runner wrapped the agent in `timeout`


R, U = NetworkMode.RESTRICTED_PROVIDER, NetworkMode.UNRESTRICTED
COMBOS = [
    Combo("astra-pi", "run-swebench-task-astra.sh", "pi", "openai", "gpt-6-astra", "medium", R, True),
    Combo("astra-opencode", "run-swebench-task-opencode-astra.sh", "opencode", "openai", "gpt-6-astra", "medium", R, True),
    Combo("mini-pi", "run-swebench-task.sh", "pi", "openai", "gpt-5.4-mini", "medium", U, False),
    Combo("mini-opencode", "run-swebench-task-opencode.sh", "opencode", "openai", "gpt-5.4-mini", "medium", R, False),
    Combo("qwen-pi", "run-swebench-task-bme-qwen.sh", "pi", "bme-lab", QWEN_MODEL, None, R, True),
    Combo("qwen-opencode", "run-swebench-task-opencode-bme-qwen.sh", "opencode", "bme-lab", QWEN_MODEL, None, R, False),
]  # fmt: skip

# Frozen-runner properties the normalized model does NOT reproduce (documented, not hidden):
HISTORICAL = {
    "no-timeout runners": {
        "mini-pi",
        "mini-opencode",
        "qwen-opencode",
    },  # plan always has a timeout
    "unrestricted network": {"mini-pi"},  # representable (NetworkMode.UNRESTRICTED)
}


def script_text(combo: Combo) -> str:
    return (ROOT / "scripts" / combo.script).read_text(encoding="utf-8")


def script_argv(text: str) -> list[str]:
    m = re.search(
        r'exec (/opt/pi-node_modules/\.bin/pi|/usr/local/bin/opencode)(.*?)"\$PROMPT"', text, re.S
    )
    assert m, "exec block not found"
    tokens = shlex.split(m.group(1) + m.group(2).replace("\\\n", " "))
    return [*tokens, PROMPT_PLACEHOLDER]


def historical_qwen_provider() -> ProviderSpec:
    """Build the generic provider definition from the frozen OpenCode config (read at test
    time, so the endpoint is never copied into new source or fixtures)."""
    cfg = json.loads((ROOT / "configs/opencode-bme-qwen-v1.json").read_text())
    ((pid, p),) = cfg["provider"].items()
    ((mid, m),) = p["models"].items()
    key_env = re.fullmatch(r"\{env:(\w+)\}", p["options"]["apiKey"]).group(1)
    pi_cfg = json.loads((ROOT / "configs/pi-bme-qwen-v1-models.json").read_text())
    (pi_model,) = next(iter(pi_cfg["providers"].values()))["models"]
    return ProviderSpec(
        id=pid,
        type="openai-compatible",
        display_name=p["name"],
        base_url=p["options"]["baseURL"],
        api_key_env=key_env,
        models={
            mid: ModelInfo(
                display_name=m["name"],
                context_window=m["limit"]["context"],
                max_output_tokens=m["limit"]["output"],
                reasoning_supported=pi_model["reasoning"],  # frozen Pi config states this
            )
        },
    )


def resolved(combo: Combo):
    providers = {"bme-lab": historical_qwen_provider()} if combo.provider == "bme-lab" else {}
    return resolve_model(f"{combo.provider}/{combo.model}", combo.reasoning, providers, {})


def squid_host(text: str) -> str:
    conf = re.search(r'SQUID_CONFIG="\$HARNESS_DIR/(configs/[^"]+)"', text).group(1)
    return re.search(r"acl \w+ dstdomain (\S+)", (ROOT / conf).read_text()).group(1)


IDS = [c.id for c in COMBOS]


@pytest.mark.parametrize("combo", COMBOS, ids=IDS)
def test_argv_matches_frozen_runner(combo):
    inv = get_harness(combo.harness).build_invocation(resolved(combo))
    assert inv.argv == script_argv(script_text(combo))


@pytest.mark.parametrize("combo", COMBOS, ids=IDS)
def test_secret_env_name_matches_frozen_runner(combo):
    inv = get_harness(combo.harness).build_invocation(resolved(combo))
    forwarded = re.search(r"-e (\w+_API_KEY) \\", script_text(combo)).group(1)
    assert inv.secret_env == [forwarded]


@pytest.mark.parametrize("combo", COMBOS, ids=IDS)
def test_container_environment_matches_frozen_runner(combo):
    text = script_text(combo)
    env = get_harness(combo.harness).build_invocation(resolved(combo)).env
    if combo.harness == "pi":
        assert env == {"PATH": re.search(r'-e PATH="([^"]+)"', text).group(1)}
    else:
        assert env == {"OPENCODE_CONFIG": re.search(r'-e OPENCODE_CONFIG="([^"]+)"', text).group(1)}


@pytest.mark.parametrize("combo", COMBOS, ids=IDS)
def test_config_files_match_frozen_configs(combo):
    text = script_text(combo)
    files = get_harness(combo.harness).build_invocation(resolved(combo)).config_files
    cfg = lambda name: json.loads((ROOT / "configs" / name).read_text())  # noqa: E731

    if combo.harness == "pi" and combo.provider == "openai":
        assert files == []
        assert "pi-models.json" not in text  # the frozen runner mounted no Pi config
    elif combo.harness == "pi":
        (f,) = files
        assert f.container_path in text  # frozen runner copies its models.json to this path
        assert json.loads(f.content) == cfg("pi-bme-qwen-v1-models.json")
    else:
        (f,) = files
        assert f.container_path in text and "/usr/local/bin/opencode" in text
        frozen = (
            "opencode-bme-qwen-v1.json"
            if combo.provider == "bme-lab"
            else "opencode-swebench-policy-v1.json"
        )
        assert re.search(rf"configs/{re.escape(frozen)}", text)
        assert json.loads(f.content) == cfg(frozen)


def script_timeout(text: str) -> int | None:
    """The frozen runner's timeout: its default seconds, or None if it had no `timeout` at all."""
    if "timeout --signal" not in text:
        return None
    return int(re.search(r"AGENT_TIMEOUT_SECONDS:-(\d+)", text).group(1))


def script_network(text: str) -> NetworkMode:
    return R if ("docker network create" in text and "--internal" in text) else U


@pytest.fixture
def plan_for(task_repo, plan_probe, plan_git):
    """Build a full RunPlan for a combo with the frozen runner's own timeout and network."""

    def build(combo: Combo):
        text = script_text(combo)
        providers = {"bme-lab": historical_qwen_provider()} if combo.provider == "bme-lab" else {}
        return build_run_plan(
            benchmark=SWEBenchMultilingual(task_repo, plan_git),
            task_id="babel__babel-15445",
            harness_name=combo.harness,
            model_ref=f"{combo.provider}/{combo.model}",
            reasoning=combo.reasoning,
            timeout_seconds=script_timeout(text),
            network_mode=script_network(text),
            providers=providers,
            probe=plan_probe,
            git=plan_git,
        ).plan

    return build


@pytest.mark.parametrize("combo", COMBOS, ids=IDS)
def test_network_intent_matches_frozen_runner(combo, plan_for):
    text = script_text(combo)
    assert script_network(text) is combo.network
    plan = plan_for(combo)
    assert plan.network.mode is combo.network
    if combo.network is R:
        assert plan.network.allowed_hosts == [squid_host(text)]  # the one host Squid allowed
    else:
        assert plan.network.allowed_hosts == [] and "squid" not in text.lower()


@pytest.mark.parametrize("combo", COMBOS, ids=IDS)
def test_timeout_matches_frozen_runner_including_no_timeout(combo, plan_for):
    text = script_text(combo)
    expected = 3600 if combo.has_timeout else None
    assert script_timeout(text) == expected
    plan = plan_for(combo)
    assert plan.timeout_seconds == expected
    assert json.loads(plan.model_dump_json())["timeout_seconds"] == expected  # null when none


def test_exactly_the_three_historical_no_timeout_treatments_have_none(plan_for):
    none_ids = {c.id for c in COMBOS if plan_for(c).timeout_seconds is None}
    assert none_ids == {"mini-pi", "mini-opencode", "qwen-opencode"}


@pytest.mark.parametrize("combo", COMBOS, ids=IDS)
def test_full_plan_invocation_equals_frozen_runner(combo, plan_for):
    plan = plan_for(combo)
    assert plan.invocation.argv == script_argv(script_text(combo))
    assert plan.model.reasoning == combo.reasoning


@pytest.mark.parametrize("combo", COMBOS, ids=IDS)
def test_output_files_match_frozen_runner(combo):
    text = script_text(combo)
    out = get_harness(combo.harness).build_invocation(resolved(combo)).output
    assert f'"$RUN_DIR/{out.stdout_file}"' in text
    assert f'"$RUN_DIR/{out.stderr_file}"' in text


@pytest.mark.parametrize("combo", COMBOS, ids=IDS)
def test_prompt_preamble_matches_frozen_runner(combo):
    block = re.search(r"\n\{\n([^{}]*?)\n\} > \"\$PROMPT_FILE\"", script_text(combo)).group(1)
    echoed = [ln.strip() for ln in block.splitlines() if ln.strip().startswith("echo")]
    lines = [re.fullmatch(r'echo\s*"?(.*?)"?', e).group(1) for e in echoed]
    assert "\n".join(lines) + "\n" == PREAMBLE  # echo lines, blank echo, then `cat statement`


def test_all_six_combinations_are_covered():
    assert len(COMBOS) == 6
    assert {c.harness for c in COMBOS} == {"pi", "opencode"}
    assert {c.model for c in COMBOS} == {"gpt-6-astra", "gpt-5.4-mini", QWEN_MODEL}


def test_historical_only_treatments_are_explicit():
    # verification-v1 is a TWO-pass treatment (solve, then a repair pass with a different
    # prompt); a single RunPlan describes one agent invocation, so it is not representable yet.
    verification = (ROOT / "scripts/run-swebench-task-verification.sh").read_text()
    assert "verification_prompt" in verification and "verification-trace.jsonl" in verification
    # the frozen BME Pi runner provisions models.json by mount+cp; the plan names the final path
    bme_pi = (ROOT / "scripts/run-swebench-task-bme-qwen.sh").read_text()
    assert "cp /tmp/pi-models.json /root/.pi/agent/models.json" in bme_pi
