"""Scan a raw-results tree for historical SWE-bench run directories."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from agent_harness_lab.artifacts.legacy import LegacyFormatError, load_run
from agent_harness_lab.artifacts.models import LoadIssue, RunRecord

SWEBENCH_SUBDIR = "swebench"


@dataclass
class DiscoveryResult:
    records: list[RunRecord] = field(default_factory=list)
    issues: list[LoadIssue] = field(default_factory=list)


def discover_runs(raw_dir: Path) -> DiscoveryResult:
    """Load every ``<raw_dir>/swebench/<task>/<run>/`` directory, read-only.

    The legacy ``js-bugfix-*`` study uses a different schema and is out of scope.
    """
    result = DiscoveryResult()
    root = raw_dir / SWEBENCH_SUBDIR
    if not root.is_dir():
        return result
    for run_dir in sorted(p for p in root.glob("*/*") if p.is_dir()):
        try:
            result.records.append(load_run(run_dir))
        except LegacyFormatError as exc:
            result.issues.append(LoadIssue(path=run_dir, reason=str(exc)))
    return result
