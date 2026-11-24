import json

import pytest

from agent_harness_lab.benchmarks.swebench_multilingual import SWEBenchMultilingual
from agent_harness_lab.providers import load_providers_file
from agent_harness_lab.runplan.builder import PlanError, build_run_plan
from agent_harness_lab.runplan.models import DEFAULT_TIMEOUT_SECONDS, NetworkMode
from agent_harness_lab.runplan.prompt import PREAMBLE, build_agent_prompt
from tests.benchmark_fixtures import COMMIT, FakeGit
from tests.plan_fixtures import AHL_COMMIT, PROJECT_ROOT, SECRET


@pytest.fixture
def make_plan(task_repo, plan_probe, plan_git, providers_file):
    benchmark = SWEBenchMultilingual(task_repo, plan_git)
    providers = load_providers_file(providers_file)

    def build(**overrides):
        kwargs = dict(
            benchmark=benchmark,
            task_id="babel__babel-15445",
            harness_name="pi",
            model_ref="openai/gpt-6-astra",
            reasoning="medium",
            timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
            network_mode=NetworkMode.RESTRICTED_PROVIDER,
            providers=providers,
            probe=plan_probe,
            git=plan_git,
        )
        kwargs.update(overrides)
        return build_run_plan(**kwargs)

    return build


# ---- content ---------------------------------------------------------------------
def test_plan_contents(make_plan):
    r = make_plan()
    p = r.plan
    assert p.schema_version == "ahl-runplan-v0"
    assert p.benchmark == "swebench-multilingual"
    assert (p.task.instance_id, p.task.repo, p.task.base_commit) == (
        "babel__babel-15445",
        "babel/babel",
        "a" * 40,
    )
    assert p.task.docker_image.endswith("babel-15445:latest")
    assert p.task.task_repository_commit == COMMIT and p.task.task_repository_dirty is False
    assert p.workdir == "/testbed"
    assert p.harness.display_name == "Pi" and p.harness.version == "1.0.2"
    assert p.harness.executable_path == "/fake/pi" and p.harness.discovery_source == "PATH"
    assert p.harness.expected_version == "1.0.2"
    assert (p.model.provider, p.model.model, p.model.reasoning) == (
        "openai",
        "gpt-6-astra",
        "medium",
    )
    assert p.provider.type == "openai" and p.provider.api_key_env == "OPENAI_API_KEY"
    assert p.required_secret_env == ["OPENAI_API_KEY"]
    assert p.timeout_seconds == 3600
    assert (p.ahl.commit, p.ahl.dirty) == (AHL_COMMIT, False)
    assert r.warnings == []


def test_prompt_construction_matches_frozen_runners():
    assert PREAMBLE == (
        "Fix the issue described below in this repository.\n"
        "Run relevant tests to verify your changes.\n\n"
    )
    # `$(cat prompt.txt)` strips ALL trailing newlines
    assert build_agent_prompt("Bug text.\n\n\n") == PREAMBLE + "Bug text."


def test_plan_records_prompt_hash_not_the_statement(make_plan):
    p = make_plan().plan
    assert p.prompt.preamble == PREAMBLE and len(p.prompt.delivered_sha256) == 64
    assert p.prompt.delivered_chars == len(PREAMBLE + "Babel bug [x] é")
    assert "Babel bug" not in p.model_dump_json()


def test_plan_json_roundtrip(make_plan):
    plan = make_plan().plan
    data = json.loads(plan.model_dump_json())
    assert data["invocation"]["argv"][-1] == "<<AHL_PROMPT>>"
    assert type(plan).model_validate(data) == plan


def test_plan_is_immutable(make_plan):
    with pytest.raises(Exception, match="frozen"):
        make_plan().plan.timeout_seconds = 1


# ---- secrets ------------------------------------------------------------------------
def test_no_secret_values_anywhere(make_plan, plan_probe):
    plan_probe.env["OPENAI_API_KEY"] = SECRET
    r = make_plan()
    blob = r.model_dump_json()
    assert SECRET not in blob and "sk-" not in blob
    assert r.plan.required_secret_env == ["OPENAI_API_KEY"]
    assert r.secret_env_present == {"OPENAI_API_KEY": True}


def test_missing_api_key_is_a_warning_not_an_error(make_plan, plan_probe):
    del plan_probe.env["OPENAI_API_KEY"]
    r = make_plan()
    assert r.secret_env_present == {"OPENAI_API_KEY": False}
    assert any("OPENAI_API_KEY is not set" in w for w in r.warnings)


# ---- timeout --------------------------------------------------------------------------
def test_timeout_default_and_override(make_plan):
    assert DEFAULT_TIMEOUT_SECONDS == 3600
    assert make_plan().plan.timeout_seconds == 3600
    assert make_plan(timeout_seconds=120).plan.timeout_seconds == 120


@pytest.mark.parametrize("bad", [0, -5])
def test_invalid_timeout(make_plan, bad):
    with pytest.raises(PlanError, match="positive"):
        make_plan(timeout_seconds=bad)


# ---- network --------------------------------------------------------------------------
def test_restricted_provider_network_openai(make_plan):
    n = make_plan().plan.network
    assert n.mode is NetworkMode.RESTRICTED_PROVIDER and n.allowed_hosts == ["api.openai.com"]


def test_restricted_provider_network_openai_compatible(make_plan):
    r = make_plan(model_ref="example-lab/Org/Model-7B-FP8", reasoning=None)
    assert r.plan.network.allowed_hosts == ["llm.example.org"]
    assert r.plan.provider.base_url == "https://llm.example.org/v1"
    assert r.plan.required_secret_env == ["EXAMPLE_LAB_API_KEY"]


def test_unrestricted_network_is_representable_and_warned(make_plan):
    r = make_plan(network_mode=NetworkMode.UNRESTRICTED)
    assert r.plan.network.mode is NetworkMode.UNRESTRICTED and r.plan.network.allowed_hosts == []
    assert any("unrestricted" in w for w in r.warnings)


# ---- harness discovery and versions ---------------------------------------------------
def test_missing_harness_executable_is_an_error(make_plan, plan_probe):
    del plan_probe.executables["pi"]
    with pytest.raises(PlanError, match="not found"):
        make_plan()


def test_project_local_harness_is_used_when_not_on_path(make_plan, plan_probe):
    del plan_probe.executables["opencode"]
    local = f"{PROJECT_ROOT}/node_modules/.bin/opencode"
    plan_probe.executable_files.add(local)
    plan_probe.commands[(local, "--version")] = (0, "opencode v2.2.0")
    r = make_plan(harness_name="opencode")
    assert (r.plan.harness.executable_path, r.plan.harness.discovery_source) == (
        local,
        "project-local",
    )
    assert r.plan.harness.version == "2.2.0"


def test_path_harness_takes_precedence_over_project_local(make_plan, plan_probe):
    plan_probe.executable_files.add(f"{PROJECT_ROOT}/node_modules/.bin/pi")
    assert make_plan().plan.harness.discovery_source == "PATH"


def test_undetectable_version_is_an_error(make_plan, plan_probe):
    plan_probe.commands[("/fake/pi", "--version")] = (1, "boom")
    with pytest.raises(PlanError, match="could not detect the version"):
        make_plan()


def test_version_mismatch_is_a_warning_with_both_versions_recorded(make_plan, plan_probe):
    plan_probe.commands[("/fake/pi", "--version")] = (0, "1.0.9")
    r = make_plan()
    assert r.plan.harness.version == "1.0.9" and r.plan.harness.expected_version == "1.0.2"
    assert any("1.0.9" in w and "pin 1.0.2" in w for w in r.warnings)


# ---- validation ----------------------------------------------------------------------
def test_unknown_harness(make_plan):
    with pytest.raises(PlanError, match="unknown harness"):
        make_plan(harness_name="aider")


@pytest.mark.parametrize("ref", ["gpt-6", "nope/model", "openai/"])
def test_bad_model_refs(make_plan, ref):
    with pytest.raises(PlanError):
        make_plan(model_ref=ref)


def test_invalid_reasoning(make_plan):
    with pytest.raises(PlanError, match="invalid reasoning"):
        make_plan(reasoning="Not Valid")
    with pytest.raises(PlanError, match="Pi does not support"):
        make_plan(reasoning="ultra")
    make_plan(harness_name="opencode", reasoning="ultra")  # OpenCode variants are model-defined


def test_missing_task_and_unreadable_provenance(make_plan, task_repo, plan_probe):
    from agent_harness_lab.benchmarks import TaskNotFound

    with pytest.raises(TaskNotFound):
        make_plan(task_id="nope__nope-1")


# ---- provenance ----------------------------------------------------------------------
def test_ahl_dirty_state_is_recorded_and_warned(make_plan, plan_git):
    plan_git.set_repo(PROJECT_ROOT, AHL_COMMIT, True)
    r = make_plan()
    assert r.plan.ahl.dirty is True
    assert any("AHL working tree has uncommitted changes" in w for w in r.warnings)


def test_ahl_commit_unknown_without_a_project_root(make_plan, plan_probe):
    plan_probe.root = None
    r = make_plan()
    assert (r.plan.ahl.commit, r.plan.ahl.dirty) == (None, None)
    assert any("AHL repository commit unknown" in w for w in r.warnings)


def test_task_repo_dirty_state_is_recorded_and_warned(task_repo, plan_probe, providers_file):
    git = FakeGit(COMMIT, dirty=True)
    git.set_repo(PROJECT_ROOT, AHL_COMMIT, False)
    r = build_run_plan(
        benchmark=SWEBenchMultilingual(task_repo, git),
        task_id="babel__babel-15445",
        harness_name="pi",
        model_ref="openai/gpt-6-astra",
        reasoning="medium",
        timeout_seconds=3600,
        network_mode=NetworkMode.RESTRICTED_PROVIDER,
        providers={},
        probe=plan_probe,
        git=git,
    )
    assert r.plan.task.task_repository_dirty is True
    assert any("task repository has uncommitted changes" in w for w in r.warnings)


def test_both_harnesses_plan_from_the_same_inputs(make_plan):
    pi = make_plan(harness_name="pi").plan
    oc = make_plan(harness_name="opencode").plan
    assert pi.task == oc.task and pi.model == oc.model and pi.prompt == oc.prompt
    assert pi.invocation.argv[0] != oc.invocation.argv[0]


# ---- timeout semantics: positive int = limit, None = no timeout -----------------------
def test_timeout_none_means_no_timeout(make_plan):
    r = make_plan(timeout_seconds=None)
    assert r.plan.timeout_seconds is None
    assert any("no execution timeout" in w for w in r.warnings)


def test_timeout_none_serializes_as_json_null_and_roundtrips(make_plan):
    plan = make_plan(timeout_seconds=None).plan
    data = json.loads(plan.model_dump_json())
    assert "timeout_seconds" in data and data["timeout_seconds"] is None
    assert type(plan).model_validate(data).timeout_seconds is None
    assert (
        json.loads(make_plan(timeout_seconds=3600).plan.model_dump_json())["timeout_seconds"]
        == 3600
    )


def test_timeout_none_is_not_confused_with_zero(make_plan):
    with pytest.raises(PlanError, match="positive"):
        make_plan(timeout_seconds=0)
    assert make_plan(timeout_seconds=None).plan.timeout_seconds is None


def test_human_rendering_of_timeout(make_plan):
    from rich.console import Console

    from agent_harness_lab.reporting.runplan import render_plan

    def render(timeout):
        console = Console(record=True, width=200, highlight=False)
        render_plan(make_plan(timeout_seconds=timeout), console)
        return console.export_text()

    assert "Timeout:                3600 s" in render(3600)
    assert "Timeout:                none" in render(None)


# ---- reasoning requested vs capability known ------------------------------------------
def test_explicit_reasoning_support_true_is_recorded_without_warning(make_plan):
    r = make_plan(model_ref="example-lab/Org/Reasoner", reasoning="high", harness_name="opencode")
    assert r.plan.model_capabilities.reasoning_supported is True
    assert r.plan.model.reasoning == "high"
    assert not any("reasoning support" in w for w in r.warnings)


def test_builtin_models_with_frozen_evidence_are_known_to_support_reasoning(make_plan):
    for model in ("gpt-6-astra", "gpt-5.4-mini"):
        r = make_plan(model_ref=f"openai/{model}", reasoning="medium")
        assert r.plan.model_capabilities.reasoning_supported is True
        assert not any("reasoning support" in w for w in r.warnings)


def test_explicit_reasoning_support_false_fails_when_reasoning_requested(make_plan):
    with pytest.raises(PlanError, match="not supporting reasoning.*'medium'"):
        make_plan(model_ref="example-lab/Org/Plain", reasoning="medium", harness_name="opencode")


def test_explicit_reasoning_support_false_is_fine_without_a_request(make_plan):
    r = make_plan(model_ref="example-lab/Org/Plain", reasoning=None)
    assert r.plan.model_capabilities.reasoning_supported is False
    assert r.plan.model.reasoning is None
    assert not any("reasoning" in w for w in r.warnings)


def test_unknown_reasoning_support_is_preserved_and_warned(make_plan):
    r = make_plan(model_ref="example-lab/Org/Model-7B-FP8", reasoning="medium")
    assert r.plan.model_capabilities.reasoning_supported is None
    assert r.plan.model.reasoning == "medium"  # never silently dropped
    assert "--thinking" in r.plan.invocation.argv  # still reaches the harness
    assert r.plan.invocation.argv[r.plan.invocation.argv.index("--thinking") + 1] == "medium"
    warning = next(w for w in r.warnings if "reasoning support" in w)
    assert "unknown" in warning and "'medium'" in warning and "unverified" in warning


def test_unknown_support_for_unlisted_openai_model_is_warned_not_assumed(make_plan):
    r = make_plan(model_ref="openai/some-future-model", reasoning="low")
    assert r.plan.model_capabilities.reasoning_supported is None
    assert any("reasoning support for openai/some-future-model is unknown" in w for w in r.warnings)


def test_unknown_support_without_a_requested_level_does_not_warn(make_plan):
    r = make_plan(model_ref="openai/some-future-model", reasoning=None)
    assert r.plan.model_capabilities.reasoning_supported is None
    assert not any("reasoning" in w for w in r.warnings)


def test_pi_level_validation_is_kept_regardless_of_capability(make_plan):
    with pytest.raises(PlanError, match="Pi does not support reasoning level 'ultra'"):
        make_plan(model_ref="example-lab/Org/Reasoner", reasoning="ultra")


def test_opencode_variant_handling_is_kept(make_plan):
    r = make_plan(model_ref="example-lab/Org/Reasoner", reasoning="ultra", harness_name="opencode")
    assert "example-lab/Org/Reasoner#ultra" in r.plan.invocation.argv
