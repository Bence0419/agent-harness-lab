"""Read-only git probe for task-repository provenance."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Protocol


class GitProbe(Protocol):
    def head_commit(self, repo: Path) -> str | None:
        """Full commit hash of HEAD, or None if ``repo`` is not a readable git repository."""
        ...

    def is_dirty(self, repo: Path) -> bool | None:
        """True if ``git status --porcelain`` is non-empty (untracked files count, as in the
        frozen runners); None if it cannot be determined."""
        ...


class GitHead:
    def head_commit(self, repo: Path) -> str | None:
        try:
            proc = subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        commit = proc.stdout.strip()
        return commit if proc.returncode == 0 and commit else None

    def is_dirty(self, repo: Path) -> bool | None:
        try:
            proc = subprocess.run(
                ["git", "-C", str(repo), "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if proc.returncode != 0:
            return None
        return bool(proc.stdout.strip())
