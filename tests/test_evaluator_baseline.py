"""The evaluator image must be at the task's baseline before ANY model-attributable verdict is
trusted, and the evaluator-built image must not pile up on disk."""

import json
import pathlib

import pytest

from agent_harness_lab.artifacts.hashing import sha256_file
from agent_harness_lab.artifacts.models import EvaluationState
from agent_harness_lab.evaluation.baseline import (
    REASON_DIRTY,
    REASON_MISMATCH,
    REASON_UNVERIFIED,
    verify_baseline,
)
from agent_harness_lab.evaluation.image_guard import EVALUATOR_IMAGE_NAMESPACE, SNAPSHOT_NAMESPACE
from agent_harness_lab.evaluation.models import Classification
from agent_harness_lab.evaluation.service import run_evaluation
from agent_harness_lab.evaluation.swebench_multilingual import SWEBenchMultilingualEvaluator
from agent_harness_lab.execution.models import DockerError
from tests.eval_fixtures import (
    TASK,
    TASK_IMAGE,
    FakeEvaluatorRunner,
    make_ahl_run,
    make_ctx,
    make_python,
)
from tests.fake_docker import FakeDocker

ADAPTER = SWEBenchMultilingualEvaluator()
BASE = "a" * 40  # the synthetic task's base_commit (FakeDocker's default /testbed HEAD)
ORIGINAL = "sha256:" + "a" * 64
REBUILT = "sha256:" + "b" * 64
APPLY_FAILED = ">>>>> Patch Apply Failed:\nerror: patch failed\n"


def docker_with_original(**kw) -> FakeDocker:
    d = FakeDocker(images={TASK_IMAGE}, **kw)
    d.image_ids[TASK_IMAGE] = ORIGINAL
    return d


@pytest.fixture
def run_env(tmp_path, task_repo):
    return {
        "run_dir": make_ahl_run(tmp_path, tmp_path / "raw"),
        "repo": task_repo,
        "python": make_python(tmp_path),
        "tmp": tmp_path,
    }


def evaluate(run_env, docker, runner=None, **kw):
    runner = runner or FakeEvaluatorRunner(docker=docker, rebuild_to=REBUILT)
    ctx = make_ctx(
        run_env["run_dir"], run_env["repo"], run_env["python"], runner, docker=docker, **kw
    )
    return run_evaluation(ADAPTER, ctx)[1]


def baseline_of(docker, **state) -> FakeDocker:
    docker.baselines[REBUILT] = state
    return docker


def verify(docker, image_id=REBUILT):
    return verify_baseline(docker, image_id, BASE, workdir="/testbed", label="t1")


# ---- the check itself -------------------------------------------------------------------------
def test_expected_commit_and_clean_worktree_is_verified():
    b = verify(FakeDocker())
    assert b.verification_status == "verified" and b.reason is None and b.error is None
    assert (b.actual_base_commit, b.expected_base_commit) == (BASE, BASE)
    assert b.worktree_clean is True and b.testbed_exists is True and b.is_git_repository is True


def test_wrong_base_commit_is_a_mismatch():
    b = verify(baseline_of(FakeDocker(), head="f" * 40))
    assert b.verification_status == "mismatch" and b.reason == REASON_MISMATCH
    assert b.actual_base_commit == "f" * 40 and b.worktree_clean is True and "f" * 12 in b.error


def test_dirty_baseline_is_not_verified():
    b = verify(baseline_of(FakeDocker(), porcelain=" M lib/x.js\n"))
    assert b.verification_status == "dirty" and b.reason == REASON_DIRTY
    assert b.actual_base_commit == BASE and b.worktree_clean is False


def test_missing_testbed_is_unverified():
    b = verify(baseline_of(FakeDocker(), exists=False))
    assert b.verification_status == "unverified" and b.reason == REASON_UNVERIFIED
    assert (
        b.testbed_exists is False and "does not exist" in b.error and b.actual_base_commit is None
    )


def test_not_a_git_repository_is_unverified():
    b = verify(baseline_of(FakeDocker(), git=False))
    assert b.verification_status == "unverified" and b.is_git_repository is False


def test_git_inspection_failure_is_unverified_never_assumed_good():
    b = verify(baseline_of(FakeDocker(), git_error=True))
    assert b.verification_status == "unverified" and b.worktree_clean is None


def test_docker_failure_while_inspecting_is_unverified():
    d = FakeDocker(fail_on={"create_container": DockerError("no space left")})
    b = verify(d)
    assert b.verification_status == "unverified" and "no space left" in b.error


def test_no_evaluator_image_id_is_unverified():
    b = verify(FakeDocker(), image_id=None)
    assert b.verification_status == "unverified" and "no evaluator image id" in b.error


def test_the_exact_image_id_is_inspected_in_a_throwaway_network_less_container():
    d = FakeDocker()
    specs = []
    orig = d.create_container
    d.create_container = lambda spec: (specs.append(spec), orig(spec))[1]
    verify(d, image_id=REBUILT)
    (spec,) = specs
    assert spec.image == REBUILT and spec.network == "none" and spec.mounts == []
    assert spec.pass_env == [] and spec.labels["ahl.managed"] == "true"
    assert d.containers == {}  # removed afterwards
    assert d.ops().index("create_container") < d.ops().index("remove")


def test_the_inspection_container_is_removed_even_when_a_check_fails():
    d = baseline_of(FakeDocker(), git_error=True)
    verify(d)
    assert "remove" in d.ops() and d.containers == {}


# ---- classification depends on the verified baseline ------------------------------------------
def test_pass_with_a_valid_baseline_remains_pass(run_env):
    doc = evaluate(run_env, docker_with_original())
    assert doc.state is EvaluationState.RESOLVED and doc.resolved is True
    assert doc.evaluator_image_baseline.verification_status == "verified" and doc.reason is None


def test_fail_with_a_valid_baseline_remains_fail(run_env):
    d = docker_with_original()
    doc = evaluate(
        run_env, d, FakeEvaluatorRunner(docker=d, rebuild_to=REBUILT, bucket="unresolved_ids")
    )
    assert doc.state is EvaluationState.UNRESOLVED and doc.resolved is False


def test_patch_apply_failure_with_a_valid_baseline_remains_a_model_fail(run_env):
    d = docker_with_original()
    runner = FakeEvaluatorRunner(
        docker=d, rebuild_to=REBUILT, bucket="error_ids", log_text=APPLY_FAILED
    )
    doc = evaluate(run_env, d, runner)
    assert doc.state is EvaluationState.PATCH_APPLY_FAILED and doc.resolved is False
    assert doc.classification is Classification.BENCHMARK_FAIL


def test_patch_apply_failure_on_a_mismatched_baseline_is_not_counted_against_the_model(run_env):
    d = baseline_of(docker_with_original(), head="f" * 40)
    runner = FakeEvaluatorRunner(
        docker=d, rebuild_to=REBUILT, bucket="error_ids", log_text=APPLY_FAILED
    )
    doc = evaluate(run_env, d, runner)
    assert doc.state is EvaluationState.INFRA_FAILURE and doc.resolved is None
    assert doc.classification is Classification.EVALUATION_FAILURE
    assert doc.reason == REASON_MISMATCH and REASON_MISMATCH in doc.error
    assert doc.official_result["state_before_baseline_check"] == "patch_apply_failed"
    assert doc.official_result["log_markers"]  # the raw official evidence is still preserved


def test_an_official_pass_from_the_wrong_baseline_is_not_trusted(run_env):
    d = baseline_of(docker_with_original(), head="f" * 40)
    doc = evaluate(run_env, d)  # the official report says resolved
    assert doc.state is EvaluationState.INFRA_FAILURE and doc.resolved is None
    assert doc.official_result["state_before_baseline_check"] == "resolved"
    assert doc.reason == REASON_MISMATCH


def test_an_official_fail_from_the_wrong_baseline_is_not_trusted_either(run_env):
    d = baseline_of(docker_with_original(), head="f" * 40)
    runner = FakeEvaluatorRunner(docker=d, rebuild_to=REBUILT, bucket="unresolved_ids")
    doc = evaluate(run_env, d, runner)
    assert doc.state is EvaluationState.INFRA_FAILURE and doc.resolved is None


@pytest.mark.parametrize(
    ("state", "reason"),
    [
        (dict(porcelain=" M x\n"), REASON_DIRTY),
        (dict(exists=False), REASON_UNVERIFIED),
        (dict(git_error=True), REASON_UNVERIFIED),
    ],
)
def test_dirty_or_unverifiable_baselines_also_yield_no_verdict(run_env, state, reason):
    doc = evaluate(run_env, baseline_of(docker_with_original(), **state))
    assert doc.state is EvaluationState.INFRA_FAILURE and doc.resolved is None
    assert doc.reason == reason


def test_an_empty_patch_does_not_depend_on_the_baseline(run_env):
    d = baseline_of(docker_with_original(), head="f" * 40)
    runner = FakeEvaluatorRunner(
        docker=d, rebuild_to=REBUILT, bucket="empty_patch_ids", report="auto"
    )
    doc = evaluate(run_env, d, runner)
    assert doc.state is EvaluationState.EMPTY_PATCH and doc.resolved is False  # nothing to apply


def test_baseline_failures_are_excluded_from_the_correctness_denominator(tmp_path, task_repo):
    from agent_harness_lab.reporting.compare import build_comparison
    from agent_harness_lab.results.service import load_runs

    raw = tmp_path / "raw"
    cases = [  # (name, bucket, log, baseline state)
        ("pass-ok", "resolved_ids", None, {}),
        ("fail-ok", "error_ids", APPLY_FAILED, {}),  # apply failure, valid baseline -> counts
        ("apply-bad", "error_ids", APPLY_FAILED, {"head": "f" * 40}),  # excluded
        ("pass-bad", "resolved_ids", None, {"head": "f" * 40}),  # excluded
    ]
    for name, bucket, log, state in cases:
        d = docker_with_original()
        d.baselines[REBUILT] = state
        run_dir = make_ahl_run(tmp_path / name, raw)
        runner = FakeEvaluatorRunner(docker=d, rebuild_to=REBUILT, bucket=bucket, log_text=log)
        ctx = make_ctx(run_dir, task_repo, make_python(tmp_path / name), runner, docker=d)
        run_evaluation(ADAPTER, ctx)
    mock = load_runs(raw).records
    assert sorted(str(r.resolved) for r in mock) == ["False", "None", "None", "True"]
    other = [r.model_copy(update={"harness": "Other", "run_id": "o"}) for r in mock[:1]]
    summary = next(
        s
        for s in build_comparison([*mock, *other], ["Mock", "Other"]).summaries
        if s.harness == "Mock"
    )
    assert (summary.runs, summary.evaluated, summary.resolved) == (4, 2, 1)
    assert summary.success_rate == 0.5


def test_the_baseline_is_recorded_as_a_structured_block(run_env):
    doc = evaluate(run_env, baseline_of(docker_with_original(), head="f" * 40))
    on_disk = json.loads((run_env["run_dir"] / "evaluation.json").read_text())
    block = on_disk["evaluator_image_baseline"]
    assert set(block) >= {
        "verification_status", "image_id", "expected_base_commit", "actual_base_commit",
        "testbed_exists", "is_git_repository", "worktree_clean", "reason", "error",
    }  # fmt: skip
    assert block["image_id"] == REBUILT  # the evaluator image, not the shared tag
    assert block["expected_base_commit"] == BASE and block["actual_base_commit"] == "f" * 40
    assert on_disk["reason"] == REASON_MISMATCH and doc.reason == REASON_MISMATCH


def test_the_inspection_uses_the_evaluator_image_never_the_shared_tag(run_env):
    d = docker_with_original()
    specs = []
    orig = d.create_container
    d.create_container = lambda spec: (specs.append(spec), orig(spec))[1]
    evaluate(run_env, d)
    (spec,) = specs
    assert spec.image == REBUILT and spec.image != TASK_IMAGE and spec.network == "none"


def test_the_baseline_check_never_touches_model_output_or_evaluator_outcome(run_env):
    d = docker_with_original()
    evaluate(run_env, d)
    argvs = [e["argv"] for e in d.exec_log if e["argv"][0] == "test" or "-C" in e["argv"]]
    assert argvs and all(a[0] in ("test", "git") for a in argvs)
    joined = json.dumps(argvs)
    for forbidden in ("gold", "hints", "patch", "apply", "tests.json"):
        assert forbidden not in joined


# ---- build provenance --------------------------------------------------------------------------
def test_dockerfile_and_task_yaml_hashes_are_recorded(run_env):
    doc = evaluate(run_env, docker_with_original())
    task_dir = run_env["repo"] / "tasks" / TASK
    assert doc.evaluator_build_inputs_sha256 == {
        "Dockerfile": sha256_file(task_dir / "Dockerfile"),
        "task.yaml": sha256_file(task_dir / "task.yaml"),
    }
    assert "Dockerfile" not in doc.evaluator_inputs_sha256  # eval inputs vs build inputs


def test_hints_and_gold_are_never_read_for_provenance(run_env, monkeypatch):
    opened = []
    real = pathlib.Path.open

    def spy(self, *a, **k):
        opened.append(self.name)
        return real(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "open", spy)
    evaluate(run_env, docker_with_original())
    assert "hints.md" not in opened and "gold.patch" not in opened


# ---- evaluator image retention ------------------------------------------------------------------
def test_the_evaluator_image_is_removed_by_default_after_the_baseline_check(run_env):
    d = docker_with_original()
    doc = evaluate(run_env, d)
    assert doc.image.retention == "removed" and doc.image.evaluator_image_ref is None
    assert d.deleted_ids == [REBUILT]
    assert [r for r in d.images if r.startswith(EVALUATOR_IMAGE_NAMESPACE)] == []
    assert d.ops().index("create_container") < d.ops().index("remove_image")  # verified first
    assert doc.image.evaluator_image_id == REBUILT  # identity survives in the record


def test_keep_evaluator_image_retains_a_reference_and_records_the_request(run_env):
    d = docker_with_original()
    doc = evaluate(run_env, d, keep_evaluator_image=True)
    assert doc.image.retention == "kept" and doc.image.keep_requested is True
    assert doc.image.evaluator_image_ref in d.images and d.deleted_ids == []
    assert d.image_ids[doc.image.evaluator_image_ref] == REBUILT


def test_pre_existing_images_are_never_removed(run_env):
    d = docker_with_original()
    d.known_ids.update({"sha256:" + "d" * 64, "sha256:" + "e" * 64})  # unrelated dangling images
    before = d.list_image_ids()
    evaluate(run_env, d)
    assert before <= d.list_image_ids() | {REBUILT}  # nothing that existed is gone
    assert set(d.deleted_ids) == {REBUILT}
    assert ORIGINAL in d.list_image_ids()


def test_an_evaluator_image_identical_to_a_pre_existing_one_is_left_alone(run_env):
    d = docker_with_original()
    d.known_ids.add(REBUILT)  # the 'new' image id already existed before this evaluation
    doc = evaluate(run_env, d)
    assert doc.image.retention == "skipped-preexisting" and d.deleted_ids == []


def test_the_original_task_tag_remains_restored_in_every_variant(run_env):
    for keep in (False, True):
        d = docker_with_original()
        run_env["run_dir"] = make_ahl_run(run_env["tmp"] / f"k{keep}", run_env["tmp"] / "raw")
        evaluate(run_env, d, keep_evaluator_image=keep)
        assert d.image_ids[TASK_IMAGE] == ORIGINAL
        assert [r for r in d.images if r.startswith(SNAPSHOT_NAMESPACE)] == []


def test_restoration_failure_keeps_every_piece_of_recovery_evidence(run_env):
    class CannotRestore(FakeDocker):
        def tag_image(self, source, target):
            if target == TASK_IMAGE:
                raise DockerError("daemon refused")
            super().tag_image(source, target)

    d = CannotRestore(images={TASK_IMAGE})
    d.image_ids[TASK_IMAGE] = ORIGINAL
    doc = evaluate(run_env, d)
    assert doc.image.restoration == "failed" and doc.image.retention == "retained-for-recovery"
    assert doc.image.snapshot_ref in d.images and not doc.image.snapshot_removed  # recovery handle
    assert REBUILT in d.list_image_ids() and d.deleted_ids == []  # evaluator image kept too
    assert any("TASK IMAGE TAG NOT RESTORED" in w for w in doc.warnings)
    assert doc.evaluator_image_baseline.verification_status == "verified"  # still checked by id


def test_interrupted_evaluation_skips_the_baseline_but_still_restores_and_cleans(run_env):
    d = docker_with_original()
    runner = FakeEvaluatorRunner(docker=d, rebuild_to=REBUILT, interrupt=True)
    with pytest.raises(KeyboardInterrupt):
        evaluate(run_env, d, runner)
    doc = json.loads((run_env["run_dir"] / "evaluation.json").read_text())
    assert (
        doc["resolved"] is None
        and doc["evaluator_image_baseline"]["verification_status"] == "unverified"
    )
    assert d.image_ids[TASK_IMAGE] == ORIGINAL and ORIGINAL not in d.deleted_ids


def test_cli_exposes_keep_evaluator_image(run_env, monkeypatch):
    from typer.testing import CliRunner

    from agent_harness_lab.cli import common
    from agent_harness_lab.cli import evaluate as evaluate_cli
    from agent_harness_lab.cli.app import app
    from tests.benchmark_fixtures import FakeGit

    d = docker_with_original()
    monkeypatch.setattr(common, "docker_client", lambda: d)
    monkeypatch.setattr(
        common, "command_runner", lambda: FakeEvaluatorRunner(docker=d, rebuild_to=REBUILT)
    )
    monkeypatch.setattr(common, "environ", lambda: {})
    monkeypatch.setattr(evaluate_cli, "_git", lambda: FakeGit())
    monkeypatch.setenv("AHL_SWEBENCH_TASKS_REPO", str(run_env["repo"]))
    base = ["evaluate", str(run_env["run_dir"]), "--swebench-python", str(run_env["python"])]
    r = CliRunner().invoke(app, [*base, "--keep-evaluator-image", "--json"], env={"COLUMNS": "200"})
    assert r.exit_code == 0, r.output
    data = json.loads(r.stdout)
    assert data["image"]["keep_requested"] is True and data["image"]["retention"] == "kept"
