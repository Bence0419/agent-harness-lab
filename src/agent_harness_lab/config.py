"""Explicit setting resolution: command-line value, then environment, else unset.

There is deliberately no directory searching and no persistent user config yet:
external paths are explicit so runs stay reproducible.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

TASKS_REPO_ENV = "AHL_SWEBENCH_TASKS_REPO"

Source = Literal["cli", "env"]


@dataclass(frozen=True)
class ResolvedPath:
    path: Path | None
    source: Source | None  # None when not configured

    @property
    def configured(self) -> bool:
        return self.path is not None


def resolve_path_setting(
    cli_value: Path | None, env_var: str, environ: Mapping[str, str] | None = None
) -> ResolvedPath:
    """Precedence: 1. CLI value, 2. environment variable, 3. not configured."""
    if cli_value is not None:
        return ResolvedPath(cli_value.expanduser(), "cli")
    env = os.environ if environ is None else environ
    raw = env.get(env_var, "").strip()
    if raw:
        return ResolvedPath(Path(raw).expanduser(), "env")
    return ResolvedPath(None, None)


def resolve_tasks_repo(
    cli_value: Path | None, environ: Mapping[str, str] | None = None
) -> ResolvedPath:
    return resolve_path_setting(cli_value, TASKS_REPO_ENV, environ)
