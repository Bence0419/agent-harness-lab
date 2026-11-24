"""Agent prompt construction, as in the frozen SWE-bench runners.

The runners wrote ``<preamble><problem_statement.md>`` to prompt.txt and then passed
``"$(cat prompt.txt)"`` to the agent. Shell command substitution strips ALL trailing
newlines, so the prompt the agent actually received is that text without trailing newlines.
"""

from __future__ import annotations

import hashlib

PREAMBLE = (
    "Fix the issue described below in this repository.\n"
    "Run relevant tests to verify your changes.\n"
    "\n"
)


def build_agent_prompt(problem_statement: str) -> str:
    return (PREAMBLE + problem_statement).rstrip("\n")


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
