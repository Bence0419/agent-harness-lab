"""Patch statistics, ported from the frozen v1 extractor.

Compatibility requires ``git apply --numstat`` (its handling of renames,
quoting and binary files defines the numbers), so the git call lives behind the
small ``NumstatProbe`` interface; everything else is pure and tested with
fixture text.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict

TEST_PATH_RE = re.compile(
    r"(^|/)(?:test|tests|__tests__)(/|$)|\.(?:test|spec)\.",
    re.I,
)


class PatchStatError(Exception):
    pass


class PatchStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    files_changed: int
    test_files_changed: int
    non_test_files_changed: int
    insertions: int
    deletions: int


class NumstatProbe(Protocol):
    def numstat(self, patch: Path) -> str:
        """Return the text of ``git apply --numstat`` for ``patch``."""
        ...


class GitApplyNumstat:
    """Real probe: runs ``git apply --numstat`` (read-only; does not touch the patch).

    Two details matter for compatibility with the frozen extractor:
    * the patch path must be absolute, because the working directory is changed;
    * ``git apply`` run from a *subdirectory* of a repository silently ignores
      patch paths outside that subdirectory. The legacy script ran from the
      repository root; we run from a fresh temporary directory outside any
      repository, which likewise filters nothing.
    """

    def numstat(self, patch: Path) -> str:
        absolute = patch.resolve()
        try:
            with tempfile.TemporaryDirectory(prefix="ahl-numstat-") as neutral_cwd:
                proc = subprocess.run(
                    ["git", "apply", "--numstat", str(absolute)],
                    cwd=neutral_cwd,
                    text=True,
                    capture_output=True,
                    check=True,
                )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise PatchStatError(f"git apply --numstat failed: {exc}") from exc
        return proc.stdout


def parse_numstat(text: str) -> PatchStats:
    files: list[str] = []
    insertions = deletions = 0
    for line in text.splitlines():
        parts = line.split("\t", 2)
        if len(parts) != 3:
            raise PatchStatError(f"unexpected numstat line: {line!r}")
        added, removed, filename = parts
        files.append(filename)
        if added.isdigit():  # binary files report "-"
            insertions += int(added)
        if removed.isdigit():
            deletions += int(removed)
    test_files = sum(bool(TEST_PATH_RE.search(f)) for f in files)
    return PatchStats(
        files_changed=len(files),
        test_files_changed=test_files,
        non_test_files_changed=len(files) - test_files,
        insertions=insertions,
        deletions=deletions,
    )


def patch_stats(path: Path, probe: NumstatProbe) -> PatchStats | None:
    """Stats for ``path``; None if the patch file is missing (unavailable, not zero).

    An existing but empty patch is a genuine empty patch: all zeros, as in v1.
    """
    if not path.is_file():
        return None
    if path.stat().st_size == 0:
        return PatchStats(
            files_changed=0,
            test_files_changed=0,
            non_test_files_changed=0,
            insertions=0,
            deletions=0,
        )
    return parse_numstat(probe.numstat(path))
