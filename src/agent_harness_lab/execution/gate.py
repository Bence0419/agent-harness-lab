"""TEMPORARY pre-release safety gate for real (non-mock) harness execution.

Real Pi/OpenCode runs spend money and call external APIs, so until the toolkit has an
explicit experiment/approval workflow they require ``AHL_ENABLE_REAL_EXECUTION=1``.
Only the exact value ``1`` enables execution: ``0``, ``true``, ``yes``, ``on``, empty or
whitespace-padded values do not. There is deliberately no CLI flag that bypasses it.
Dry-run and pre-flight never need the gate. Remove this once experiment workflows exist.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

GATE_ENV = "AHL_ENABLE_REAL_EXECUTION"
GATE_ENABLED_VALUE = "1"


def real_execution_enabled(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return env.get(GATE_ENV) == GATE_ENABLED_VALUE
