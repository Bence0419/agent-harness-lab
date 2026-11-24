import pytest

from agent_harness_lab.providers import (
    ProviderError,
    ProviderSpec,
    load_providers_file,
    parse_model_ref,
    resolve_model,
    validate_reasoning,
)
from tests.plan_fixtures import PROVIDERS_YAML


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("openai/gpt-6-astra", ("openai", "gpt-6-astra")),
        ("example-lab/Org/Model-7B-FP8", ("example-lab", "Org/Model-7B-FP8")),
        ("p/a/b/c", ("p", "a/b/c")),  # only the FIRST slash separates provider from model
        ("openai/gpt-5.4-mini", ("openai", "gpt-5.4-mini")),
    ],
)
def test_model_reference_parsing_keeps_nested_model_ids(ref, expected):
    assert parse_model_ref(ref) == expected


@pytest.mark.parametrize("ref", ["gpt-6-astra", "/gpt", "openai/", "openai/  ", ""])
def test_invalid_model_references(ref):
    with pytest.raises(ProviderError, match="expected <provider>/<model>"):
        parse_model_ref(ref)


def test_reasoning_validation():
    assert validate_reasoning(None) is None
    assert validate_reasoning("medium") == "medium"
    for bad in ("Medium", "#high", "me dium", "", "1x"):
        with pytest.raises(ProviderError, match="invalid reasoning"):
            validate_reasoning(bad)


def test_openai_provider_is_built_in():
    m = resolve_model("openai/gpt-6-astra", "medium", {}, {})
    assert m.provider.type == "openai" and m.provider.api_key_env == "OPENAI_API_KEY"
    assert m.base_url is None and m.endpoint_host == "api.openai.com"
    assert (m.model.provider, m.model.model, m.model.reasoning) == (
        "openai",
        "gpt-6-astra",
        "medium",
    )


def test_openai_compatible_provider_from_file(tmp_path, providers_file):
    providers = load_providers_file(providers_file)
    m = resolve_model("example-lab/Org/Model-7B-FP8", None, providers, {})
    assert m.provider.type == "openai-compatible"
    assert m.base_url == "https://llm.example.org/v1" and m.endpoint_host == "llm.example.org"
    assert m.model.model == "Org/Model-7B-FP8" and m.model.display_name == "Example 7B"
    assert m.info.context_window == 32768


def test_base_url_env_is_resolved_at_plan_time(providers_file):
    providers = load_providers_file(providers_file)
    m = resolve_model(
        "env-lab/m", None, providers, {"ENV_LAB_BASE_URL": "https://h.example.net/v1"}
    )
    assert m.base_url == "https://h.example.net/v1" and m.endpoint_host == "h.example.net"


def test_base_url_env_missing_or_invalid(providers_file):
    providers = load_providers_file(providers_file)
    with pytest.raises(ProviderError, match="ENV_LAB_BASE_URL"):
        resolve_model("env-lab/m", None, providers, {})
    with pytest.raises(ProviderError, match="not a valid http"):
        resolve_model("env-lab/m", None, providers, {"ENV_LAB_BASE_URL": "ftp://x"})


def test_unknown_provider_lists_known_ones(providers_file):
    providers = load_providers_file(providers_file)
    with pytest.raises(ProviderError, match="unknown provider 'Qwen'.*example-lab.*openai"):
        resolve_model("Qwen/Qwen3-8B", None, providers, {})


def test_builtin_openai_cannot_be_redefined(tmp_path):
    f = tmp_path / "p.yaml"
    f.write_text(
        "providers:\n  openai:\n    type: openai-compatible\n"
        "    base_url: https://x/v1\n    api_key_env: K\n"
    )
    with pytest.raises(ProviderError, match="cannot be redefined"):
        load_providers_file(f)


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        ("type: openai-compatible\n    api_key_env: K\n", "exactly one of base_url"),
        (
            "type: openai-compatible\n    base_url: https://x/v1\n"
            "    base_url_env: B\n    api_key_env: K\n",
            "exactly one",
        ),
        (
            "type: openai-compatible\n    base_url: https://x/v1\n    api_key_env: lowercase\n",
            "environment variable NAME",
        ),
        (
            "type: openai-compatible\n    base_url: not-a-url\n    api_key_env: K\n",
            "http\\(s\\) URL",
        ),
        (
            "type: openai-compatible\n    base_url: https://u:p@x/v1\n    api_key_env: K\n",
            "credentials",
        ),
        ("type: mystery\n    base_url: https://x/v1\n    api_key_env: K\n", "type"),
    ],
)
def test_invalid_provider_definitions(tmp_path, body, fragment):
    f = tmp_path / "p.yaml"
    f.write_text(f"providers:\n  lab:\n    {body}")
    with pytest.raises(ProviderError, match=fragment):
        load_providers_file(f)


def test_secret_in_providers_file_is_rejected_without_being_echoed(tmp_path):
    secret = "sk-live-0123456789abcdef"
    f = tmp_path / "p.yaml"
    f.write_text(
        "providers:\n  lab:\n    type: openai-compatible\n    base_url: https://x/v1\n"
        f"    api_key: {secret}\n    api_key_env: {secret}\n"
    )
    with pytest.raises(ProviderError) as exc:
        load_providers_file(f)
    assert secret not in str(exc.value)
    assert "api_key" in str(exc.value)  # the offending FIELD is named, never its value


def test_malformed_providers_files(tmp_path):
    f = tmp_path / "p.yaml"
    f.write_text("a: [unclosed")
    with pytest.raises(ProviderError, match="not valid YAML"):
        load_providers_file(f)
    f.write_text("- not a mapping")
    with pytest.raises(ProviderError, match="'providers:' mapping"):
        load_providers_file(f)
    with pytest.raises(ProviderError, match="cannot read"):
        load_providers_file(tmp_path / "missing.yaml")


def test_provider_ids_cannot_contain_slash():
    with pytest.raises(ValueError, match="no '/'"):
        ProviderSpec(id="a/b", type="openai", api_key_env="K")


def test_providers_file_is_json_compatible(tmp_path):
    f = tmp_path / "p.json"
    f.write_text(
        '{"providers": {"lab": {"type": "openai-compatible", '
        '"base_url": "https://x/v1", "api_key_env": "K"}}}'
    )
    assert "lab" in load_providers_file(f)
    assert PROVIDERS_YAML  # fixture sanity


# ---- capability is tri-state and never invented ----------------------------------------
def test_model_info_capability_defaults_to_unknown():
    from agent_harness_lab.providers import ModelInfo

    assert ModelInfo().reasoning_supported is None
    assert ModelInfo(reasoning_supported=True).reasoning_supported is True
    assert ModelInfo(reasoning_supported=False).reasoning_supported is False


def test_capability_from_providers_file_is_tristate(providers_file):
    from agent_harness_lab.providers import reasoning_capability

    providers = load_providers_file(providers_file)
    cap = {
        m: reasoning_capability(resolve_model(f"example-lab/{m}", None, providers, {}))
        for m in ("Org/Reasoner", "Org/Plain", "Org/Model-7B-FP8", "Org/Never-Listed")
    }
    assert cap == {
        "Org/Reasoner": True,
        "Org/Plain": False,
        "Org/Model-7B-FP8": None,  # listed for limits only: capability still unknown
        "Org/Never-Listed": None,  # not listed at all: unknown
    }


def test_only_evidenced_builtin_models_have_known_capability():
    from agent_harness_lab.providers import OPENAI, reasoning_capability

    assert {m: i.reasoning_supported for m, i in OPENAI.models.items()} == {
        "gpt-6-astra": True,
        "gpt-5.4-mini": True,
    }
    for other in ("gpt-4o", "o3", "gpt-7", "anything-else"):
        assert reasoning_capability(resolve_model(f"openai/{other}", None, {}, {})) is None


def test_requesting_reasoning_never_changes_capability():
    from agent_harness_lab.providers import reasoning_capability

    m = resolve_model("openai/gpt-7", "high", {}, {})
    assert m.model.reasoning == "high" and reasoning_capability(m) is None
