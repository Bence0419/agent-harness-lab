"""Builders for small fixture run directories mirroring the real historical shapes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BASE = {
    "instanceId": "proj__repo-1",
    "taskRepoCommit": "6e08cbcd763c869abd7aae4bcb9e1b3991180777",
    "baseCommit": "b" * 40,
    "image": "swebench/sweb.eval.x86_64.proj_1776_repo-1:latest",
    "harnessCommit": "c" * 40,
    "harnessDirty": False,
    "startedAt": "2026-11-18T10:00:00Z",
    "finishedAt": "2026-11-18T10:02:30Z",
}

SHAPES: dict[str, dict[str, Any]] = {
    "pi-swebench-baseline": {
        **BASE,
        "piVersion": "1.0.2",
        "provider": "openai",
        "model": "gpt-5.4-mini",
        "thinking": "medium",
        "tools": "read,bash,edit,write",
        "piExitCode": 0,
    },
    "pi-swebench-reference": {
        **BASE,
        "reference": "pi-frontier-astra-v1",
        "piVersion": "1.0.2",
        "provider": "openai",
        "model": "gpt-6-astra",
        "thinking": "medium",
        "tools": "read,bash,edit,write",
        "piExitCode": 0,
    },
    "pi-swebench-verification": {
        **BASE,
        "experiment": "verification-v1",
        "base": "baseline-v1",
        "verificationPasses": 1,
        "piVersion": "1.0.2",
        "provider": "openai",
        "model": "gpt-5.4-mini",
        "thinking": "medium",
        "tools": "read,bash,edit,write",
        "piExitCode": 0,
        "solveExitCode": 0,
        "verificationExitCode": 0,
    },
    "opencode-swebench": {
        **BASE,
        "reference": "opencode-frontier-astra-v1",
        "opencodeVersion": "2.2.0",
        "agent": "build",
        "provider": "openai",
        "model": "gpt-6-astra",
        "variant": "medium",
        "nativeHarness": True,
        "networkPolicy": "openai-only-squid-v1",
        "squidImage": "ubuntu/squid@sha256:abc",
        "opencodeExitCode": 0,
    },
}


def evaluation(run_id: str, task: str, outcome: str = "resolved") -> dict[str, Any]:
    flags = {
        k: False
        for k in (
            "resolved",
            "unresolved",
            "incomplete",
            "infraFailure",
            "ambiguousFailure",
            "emptyPatch",
            "error",
        )
    }
    flags[outcome] = True
    return {
        "schemaVersion": 1,
        "instanceId": task,
        "generationRunId": run_id,
        "evaluationRunId": f"eval-{run_id}",
        **flags,
    }


def make_run(
    raw: Path,
    run_id: str,
    shape: str = "pi-swebench-reference",
    *,
    outcome: str | None = "resolved",
    remove: tuple[str, ...] = (),
    **overrides: Any,
) -> Path:
    """Write <raw>/swebench/<task>/<run_id>/{metadata,evaluation}.json."""
    meta = {**SHAPES[shape], "runId": run_id, **overrides}
    for key in remove:
        meta.pop(key, None)
    run_dir = raw / "swebench" / meta["instanceId"] / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(json.dumps(meta))
    if outcome is not None:
        (run_dir / "evaluation.json").write_text(
            json.dumps(evaluation(run_id, meta["instanceId"], outcome))
        )
    return run_dir
