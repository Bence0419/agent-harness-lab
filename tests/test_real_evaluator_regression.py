"""Opt-in: the new evaluator must reproduce the official result of existing historical runs.

Same patch + same evaluator semantics -> same resolved state. Needs:
  AHL_REAL_EVALUATOR=1, AHL_SWEBENCH_PYTHON (interpreter of the SWE-bench environment),
  AHL_SWEBENCH_TASKS_REPO, a running Docker daemon and the historical task images, plus the
  ids of the historical runs to re-score (their existing evaluation.json is the oracle):
  AHL_REGRESSION_PASS_RUN (a run that was resolved) and AHL_REGRESSION_FAIL_RUN (unresolved).
NOTE: the official evaluator rebuilds task images from the task repository, so each run
re-tags the local task image; pick runs deliberately.
Historical run directories are only READ; results go to a temporary output directory.
"""

import os
from pathlib import Path

import pytest

from agent_harness_lab.benchmarks import create_benchmark
from agent_harness_lab.benchmarks.git import GitHead
from agent_harness_lab.evaluation.base import EvaluationContext
from agent_harness_lab.evaluation.registry import get_evaluator
from agent_harness_lab.evaluation.service import load_subject, run_evaluation
from agent_harness_lab.execution.docker import DockerCli, SubprocessRunner
from agent_harness_lab.paths import default_raw_dir
from agent_harness_lab.results.service import load_runs

PYTHON = os.environ.get("AHL_SWEBENCH_PYTHON")
REPO = os.environ.get("AHL_SWEBENCH_TASKS_REPO")

pytestmark = pytest.mark.skipif(
    os.environ.get("AHL_REAL_EVALUATOR") != "1" or not (PYTHON and REPO),
    reason="set AHL_REAL_EVALUATOR=1 with AHL_SWEBENCH_PYTHON and AHL_SWEBENCH_TASKS_REPO",
)


@pytest.mark.parametrize(
    ("env_var", "expected"),
    [("AHL_REGRESSION_PASS_RUN", True), ("AHL_REGRESSION_FAIL_RUN", False)],
    ids=["historical-pass", "historical-fail"],
)
def test_new_evaluator_reproduces_the_historical_result(tmp_path, env_var, expected):
    run_id = os.environ.get(env_var)
    if not run_id:
        pytest.skip(f"set {env_var} to a historical run id")
    store = default_raw_dir()
    hist = next((r for r in load_runs(store).records if r.run_id == run_id), None)
    assert hist is not None, f"historical run not found: {run_id}"
    assert hist.resolved is expected, "the named run does not have the expected historical outcome"
    benchmark = create_benchmark("swebench-multilingual", Path(REPO), GitHead())
    ctx = EvaluationContext(
        subject=load_subject(hist.artifact_dir),
        benchmark=benchmark,
        repository=benchmark.repository_info(),
        python=Path(PYTHON),
        docker=DockerCli(),
        runner=SubprocessRunner(),
        environ=os.environ,
        output_dir=tmp_path,  # never the historical directory
    )
    docker = ctx.docker
    tag_before = docker.image_inspect(hist.docker_image).image_id
    _, doc = run_evaluation(get_evaluator("swebench-multilingual"), ctx)
    assert doc is not None, "pre-flight failed"
    assert doc.resolved is expected, (doc.state, doc.error, doc.official_result)
    assert not (hist.artifact_dir / "evaluator").exists()  # historical directory untouched
    # the shared task tag is back on EXACTLY the image it had before this evaluation
    assert docker.image_inspect(hist.docker_image).image_id == tag_before
    assert doc.image.image_id_before == tag_before == doc.image.final_image_id
    assert doc.image.restoration in ("restored", "not-needed")
    refs = docker.list_image_refs()
    assert not any(r.startswith("ahl-eval-snapshot/") for r in refs)
    assert not any(r.startswith("ahl-eval-image/") for r in refs)  # default: nothing accumulates
    # the evaluator image really was at the task baseline, and clean
    task = benchmark.get_task(hist.task)
    b = doc.evaluator_image_baseline
    assert b.verification_status == "verified", b
    assert b.actual_base_commit == b.expected_base_commit == task.base_commit
    assert b.worktree_clean is True and b.image_id == doc.image.evaluator_image_id
    assert doc.evaluator_build_inputs_sha256.keys() == {"Dockerfile", "task.yaml"}
