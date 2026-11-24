"""Verification-command heuristic, exactly as in the frozen v1 extractor.

A tool call counts as a verification call when it is a shell command whose text
contains a test-runner invocation matched by ``TEST_RE``.

Known imperfections (documented, deliberately NOT fixed in ahl-metrics-v1):
* it is a substring match, so ``grep pytest`` or ``pip install pytest`` count;
* runners not listed are missed, e.g. ``mvn test``, ``./gradlew test``,
  ``make test``, ``tox``, ``ctest``. Java tasks are therefore likely under-counted.
"""

from __future__ import annotations

import re

TEST_RE = re.compile(
    r"(?:pytest|jest|vitest|mocha|karma|cargo\s+test|go\s+test|"
    r"(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?test)",
    re.I,
)


def is_verification_command(command: str | None) -> bool:
    return bool(command) and TEST_RE.search(command) is not None
