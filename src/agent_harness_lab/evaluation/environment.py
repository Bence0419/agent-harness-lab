"""Detect the evaluator environment from a Python interpreter path (no hardcoded locations)."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

from agent_harness_lab.evaluation.models import EvaluatorProvenance
from agent_harness_lab.execution.docker import CommandRunner

_SECRET_NAME = re.compile(r"(_API_KEY|_TOKEN|_SECRET|_PASSWORD)$", re.IGNORECASE)

PROBE = (
    "import json, sys, importlib.metadata as m\n"
    "import swebench\n"
    "print(json.dumps({'executable': sys.executable, 'python': sys.version.split()[0],\n"
    "                  'swebench': m.version('swebench')}))"
)


class EvaluatorEnvironmentError(Exception):
    pass


def scrubbed_env(environ: Mapping[str, str]) -> dict[str, str]:
    """Environment for the evaluator process WITHOUT any agent credentials: evaluation calls no
    model and must not be able to depend on (or leak) API keys."""
    return {k: v for k, v in environ.items() if not _SECRET_NAME.search(k)}


def cli_path(python: Path) -> Path:
    """The evaluator's console script, installed next to the interpreter (``swebench``)."""
    return python.parent / "swebench"


def detect_environment(
    python: Path, runner: CommandRunner, environ: Mapping[str, str]
) -> EvaluatorProvenance:
    result = runner.run([str(python), "-c", PROBE], timeout=120, env=scrubbed_env(environ))
    if result.returncode != 0:
        tail = (result.stderr or result.stdout).strip().splitlines()
        raise EvaluatorEnvironmentError(
            f"the evaluator package cannot be imported with {python}: "
            f"{tail[-1] if tail else 'unknown error'}"
        )
    try:
        info = json.loads(result.stdout.strip().splitlines()[-1])
        return EvaluatorProvenance(
            python_executable=info["executable"],
            python_version=info["python"],
            evaluator_package="swebench",
            evaluator_version=info.get("swebench"),
            cli=str(cli_path(python)),
        )
    except (ValueError, IndexError, KeyError) as exc:
        raise EvaluatorEnvironmentError("unreadable evaluator environment probe output") from exc
