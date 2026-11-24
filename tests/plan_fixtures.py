"""Constants for planning tests (synthetic provider; no real endpoints or secrets)."""

from __future__ import annotations

PROJECT_ROOT = "/fake/project"
AHL_COMMIT = "a" * 40
SECRET = "sk-super-secret-value"

PACKAGE_JSON = (
    '{"dependencies": {"@earendil-works/pi-coding-agent": "1.0.2", "@opencode/cli": "2.2.0"}}'
)

PROVIDERS_YAML = """\
providers:
  example-lab:
    type: openai-compatible
    display_name: Example Lab
    base_url: https://llm.example.org/v1
    api_key_env: EXAMPLE_LAB_API_KEY
    models:
      "Org/Model-7B-FP8":
        display_name: Example 7B
        context_window: 32768
        max_output_tokens: 4096
      "Org/Reasoner":
        display_name: Example Reasoner
        context_window: 32768
        max_output_tokens: 4096
        reasoning_supported: true
      "Org/Plain":
        context_window: 8192
        max_output_tokens: 1024
        reasoning_supported: false
  env-lab:
    type: openai-compatible
    base_url_env: ENV_LAB_BASE_URL
    api_key_env: ENV_LAB_API_KEY
"""
