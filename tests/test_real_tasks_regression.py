"""Opt-in: validate the adapter against a real task checkout and the frozen manifests.

Needs AHL_REAL_DATA_REGRESSION=1 and AHL_SWEBENCH_TASKS_REPO pointing at a checkout.
Reads only task.yaml / problem_statement.md, like the adapter itself.
"""

import json
import os
from pathlib import Path

import pytest

from agent_harness_lab.benchmarks import create_benchmark
from agent_harness_lab.paths import find_project_root

TASKS_REPO = os.environ.get("AHL_SWEBENCH_TASKS_REPO")

pytestmark = pytest.mark.skipif(
    os.environ.get("AHL_REAL_DATA_REGRESSION") != "1" or not TASKS_REPO,
    reason="set AHL_REAL_DATA_REGRESSION=1 and AHL_SWEBENCH_TASKS_REPO to run",
)


def test_all_real_tasks_load_and_match_frozen_manifests():
    listing = create_benchmark("swebench-multilingual", Path(TASKS_REPO)).list_tasks()
    assert listing.problems == []
    by_id = {t.instance_id: t for t in listing.tasks}

    root = find_project_root()
    pilot = json.loads((root / "experiments/swebench/swebench-pilot-5-v1.json").read_text())
    challenge = json.loads((root / "configs/challenge-set-v1.json").read_text())
    for i in pilot["instances"]:
        t = by_id[i["instanceId"]]
        assert (t.repo, t.base_commit, t.docker_image) == (i["repo"], i["baseCommit"], i["image"])
    for ids in challenge["ecosystems"].values():
        for iid in ids:
            assert iid in by_id
