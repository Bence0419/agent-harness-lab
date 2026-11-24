"""Normalized provider/model configuration.

Design
------
* A **provider** says where a model is served and which *environment variable* holds its
  credential. Secret values are never part of any model here: only variable NAMES.
* Two provider types: ``openai`` (built in, id ``openai``) and ``openai-compatible``
  (user-defined in a providers file).
* An OpenAI-compatible provider has exactly one of ``base_url`` (literal, non-secret URL)
  or ``base_url_env`` (name of an env var holding the URL, so internal endpoints need not
  appear in files). The resolved URL is recorded in run plans.
* A **model reference** on the command line is ``<provider-id>/<model-id>``. It is split on
  the FIRST slash only, so model ids may themselves contain slashes:
  ``bme-lab/Qwen/Qwen3.6-35B-A3B-FP8`` -> provider ``bme-lab``, model ``Qwen/Qwen3.6-35B-A3B-FP8``.
  Provider ids therefore never contain ``/``.
* Reasoning is one generic field (e.g. ``medium``). Translating it into a harness's own
  syntax is the harness adapter's job; nothing harness-specific lives here.

Providers file (YAML or JSON), selected by ``--providers-file`` or ``AHL_PROVIDERS_FILE``::

    providers:
      my-lab:
        type: openai-compatible
        display_name: My Lab
        base_url_env: MY_LAB_BASE_URL      # or: base_url: https://llm.example.org/v1
        api_key_env: MY_LAB_API_KEY
        models:                             # optional per-model metadata used when rendering
          "Qwen/Qwen3.6-35B-A3B-FP8":
            display_name: Qwen 35B
            context_window: 262144
            max_output_tokens: 16384
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Protocol
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

PROVIDERS_FILE_ENV = "AHL_PROVIDERS_FILE"
OPENAI_PROVIDER_ID = "openai"
OPENAI_API_HOST = "api.openai.com"

_PROVIDER_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_REASONING = re.compile(r"^[a-z][a-z0-9_-]*$")


class EnvLike(Protocol):
    def get(self, name: str, default: str | None = None) -> str | None: ...


class ProviderError(Exception):
    """Invalid provider/model configuration. Messages never contain input values."""


class ModelInfo(BaseModel):
    """Optional per-model metadata. Never invented: absent unless the user supplies it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    display_name: str | None = None
    context_window: int | None = None
    max_output_tokens: int | None = None
    # Whether the model is KNOWN to support reasoning selection:
    #   True -> known to support it; False -> known not to; None -> unknown (the default).
    # This describes capability only; the level a run requests is ModelSpec.reasoning.
    reasoning_supported: bool | None = None


class ProviderSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    type: Literal["openai", "openai-compatible"]
    display_name: str | None = None
    base_url: str | None = None
    base_url_env: str | None = None
    api_key_env: str
    models: dict[str, ModelInfo] = {}

    @field_validator("id")
    @classmethod
    def _id(cls, v: str) -> str:
        if not _PROVIDER_ID.match(v):
            raise ValueError("provider id must match [a-z0-9][a-z0-9_-]* (no '/')")
        return v

    @field_validator("api_key_env", "base_url_env")
    @classmethod
    def _env_name(cls, v: str | None) -> str | None:
        if v is not None and not _ENV_NAME.match(v):
            # Also catches a pasted secret: a key is never a valid variable NAME.
            raise ValueError("must be an environment variable NAME (A-Z, 0-9, _)")
        return v

    @model_validator(mode="after")
    def _endpoint(self) -> ProviderSpec:
        if self.type == "openai-compatible":
            if (self.base_url is None) == (self.base_url_env is None):
                raise ValueError("openai-compatible needs exactly one of base_url / base_url_env")
        elif self.base_url is not None or self.base_url_env is not None:
            raise ValueError("the built-in openai provider does not take a base_url")
        if self.base_url is not None:
            parsed = urlparse(self.base_url)
            if parsed.scheme not in ("http", "https") or not parsed.hostname:
                raise ValueError("base_url must be an http(s) URL with a host")
            if parsed.username or parsed.password:
                raise ValueError("base_url must not embed credentials")
        return self


# Capability is recorded only where this repository has evidence. Both models below were run
# at reasoning=medium in the frozen studies and EVERY such run reported reasoning tokens
# (gpt-6-astra: 22/22 runs; gpt-5.4-mini: 21/21 runs across Pi and OpenCode). Any other model
# stays "unknown" until a user or later evidence says otherwise.
_EVIDENCED_REASONING_MODELS = ("gpt-6-astra", "gpt-5.4-mini")

OPENAI = ProviderSpec(
    id=OPENAI_PROVIDER_ID,
    type="openai",
    api_key_env="OPENAI_API_KEY",
    models={m: ModelInfo(reasoning_supported=True) for m in _EVIDENCED_REASONING_MODELS},
)


class ModelSpec(BaseModel):
    """What the user asked for. Harness-agnostic."""

    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    reasoning: str | None = None
    display_name: str | None = None


class ResolvedModel(BaseModel):
    """A ModelSpec joined with its provider and (if the provider defines it) model metadata."""

    model_config = ConfigDict(frozen=True)

    provider: ProviderSpec
    model: ModelSpec
    base_url: str | None  # resolved endpoint for openai-compatible; None for built-in openai
    info: ModelInfo | None = None

    @property
    def endpoint_host(self) -> str:
        if self.provider.type == "openai":
            return OPENAI_API_HOST
        assert self.base_url is not None
        host = urlparse(self.base_url).hostname
        assert host is not None
        return host


def reasoning_capability(model: ResolvedModel) -> bool | None:
    """True/False if known, None if unknown. Never guessed."""
    return model.info.reasoning_supported if model.info is not None else None


def parse_model_ref(ref: str) -> tuple[str, str]:
    """Split ``provider/model`` on the first slash. The model id may contain more slashes."""
    provider, sep, model = ref.partition("/")
    if not sep or not provider or not model.strip():
        raise ProviderError(
            f"invalid model reference {ref!r}: expected <provider>/<model>, "
            "e.g. openai/gpt-6-astra"
        )
    return provider, model


def validate_reasoning(reasoning: str | None) -> str | None:
    if reasoning is None:
        return None
    if not _REASONING.match(reasoning):
        raise ProviderError(
            f"invalid reasoning value {reasoning!r}: use a lowercase word such as low/medium/high"
        )
    return reasoning


def _validation_summary(prefix: str, exc: ValidationError) -> str:
    # Never echo input values: they could be a mistakenly pasted secret.
    parts = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]
    return f"{prefix}: " + "; ".join(parts)


def load_providers_file(path: Path) -> dict[str, ProviderSpec]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ProviderError(f"cannot read providers file {path}: {exc.strerror}") from exc
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        raise ProviderError(f"providers file {path} is not valid YAML/JSON") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("providers"), dict):
        raise ProviderError(f"providers file {path} must contain a 'providers:' mapping")
    out: dict[str, ProviderSpec] = {}
    for pid, body in raw["providers"].items():
        if not isinstance(body, dict):
            raise ProviderError(f"provider {pid!r} must be a mapping")
        if pid == OPENAI_PROVIDER_ID:
            raise ProviderError("the built-in 'openai' provider cannot be redefined")
        try:
            out[str(pid)] = ProviderSpec.model_validate({**body, "id": str(pid)})
        except ValidationError as exc:
            raise ProviderError(_validation_summary(f"provider {pid!r}", exc)) from exc
    return out


def resolve_model(
    ref: str,
    reasoning: str | None,
    providers: Mapping[str, ProviderSpec],
    environ: EnvLike | None = None,
) -> ResolvedModel:
    env: EnvLike = os.environ if environ is None else environ
    provider_id, model_id = parse_model_ref(ref)
    validate_reasoning(reasoning)

    known = {OPENAI_PROVIDER_ID: OPENAI, **providers}
    provider = known.get(provider_id)
    if provider is None:
        raise ProviderError(
            f"unknown provider {provider_id!r} in model reference {ref!r}; known providers: "
            f"{', '.join(sorted(known))}. Define custom providers in a providers file "
            f"(--providers-file or ${PROVIDERS_FILE_ENV})."
        )

    base_url = provider.base_url
    if provider.base_url_env is not None:
        base_url = (env.get(provider.base_url_env, "") or "").strip() or None
        if base_url is None:
            raise ProviderError(
                f"provider {provider.id!r}: environment variable {provider.base_url_env} "
                "(base_url_env) is not set"
            )
        try:
            ProviderSpec(
                id=provider.id,
                type=provider.type,
                base_url=base_url,
                api_key_env=provider.api_key_env,
            )
        except ValidationError as exc:
            raise ProviderError(
                f"provider {provider.id!r}: ${provider.base_url_env} is not a valid http(s) URL"
            ) from exc

    info = provider.models.get(model_id)
    spec = ModelSpec(
        provider=provider.id,
        model=model_id,
        reasoning=reasoning,
        display_name=(info.display_name if info else None),
    )
    return ResolvedModel(provider=provider, model=spec, base_url=base_url, info=info)
