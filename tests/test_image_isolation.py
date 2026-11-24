"""The official evaluator force-rebuilds and re-tags the shared task image. AHL pins the original
image id under a temporary reference, lets the evaluator run unchanged, records the evaluator's
image, then restores and VERIFIES the original tag. No Docker daemon is used here."""

import json

import pytest

from agent_harness_lab.artifacts.models import EvaluationState
from agent_harness_lab.evaluation.base import EvaluationError
from agent_harness_lab.evaluation.image_guard import (
    EVALUATOR_IMAGE_NAMESPACE,
    SNAPSHOT_NAMESPACE,
    ImageGuard,
)
from agent_harness_lab.evaluation.service import run_evaluation
from agent_harness_lab.evaluation.swebench_multilingual import SWEBenchMultilingualEvaluator
from agent_harness_lab.execution.models import DockerError
from tests.eval_fixtures import (
    TASK_IMAGE,
    FakeEvaluatorRunner,
    make_ahl_run,
    make_ctx,
    make_python,
)
from tests.fake_docker import FakeDocker

ADAPTER = SWEBenchMultilingualEvaluator()
ORIGINAL = "sha256:" + "a" * 64
REBUILT = "sha256:" + "b" * 64


def docker_with_original() -> FakeDocker:
    d = FakeDocker(images={TASK_IMAGE})
    d.image_ids[TASK_IMAGE] = ORIGINAL
    return d


@pytest.fixture
def run_env(tmp_path, task_repo):
    return {
        "run_dir": make_ahl_run(tmp_path, tmp_path / "raw"),
        "repo": task_repo,
        "python": make_python(tmp_path),
    }


def evaluate(run_env, docker, runner, **kw):
    ctx = make_ctx(
        run_env["run_dir"], run_env["repo"], run_env["python"], runner, docker=docker, **kw
    )
    return run_evaluation(ADAPTER, ctx)


def refs(docker, prefix):
    return sorted(r for r in docker.images if r.startswith(prefix))


# ---- guard unit behaviour ----------------------------------------------------------------------
def test_guard_pins_restores_and_cleans_up():
    d = docker_with_original()
    g = ImageGuard(d, TASK_IMAGE, "run1")
    assert g.snapshot() == ORIGINAL
    assert refs(d, SNAPSHOT_NAMESPACE)  # the original is pinned while the evaluator runs
    d.external_rebuild(TASK_IMAGE, REBUILT)  # what the evaluator does
    out = g.restore()
    assert (out.image_id_before, out.evaluator_image_id, out.final_image_id) == (
        ORIGINAL,
        REBUILT,
        ORIGINAL,
    )
    assert out.rebuilt and out.restoration == "restored" and out.snapshot_removed
    assert d.image_ids[TASK_IMAGE] == ORIGINAL
    assert refs(d, SNAPSHOT_NAMESPACE) == []  # temporary reference removed
    assert d.deleted_ids == []  # restoring deletes nothing; the evaluator image is released later


def test_without_the_guard_the_original_image_would_be_deleted_control():
    d = docker_with_original()
    d.external_rebuild(TASK_IMAGE, REBUILT)  # no snapshot
    assert d.deleted_ids == [ORIGINAL]  # proves the fake models the real hazard


def test_guard_when_the_evaluator_did_not_rebuild():
    d = docker_with_original()
    g = ImageGuard(d, TASK_IMAGE, "run1")
    g.snapshot()
    out = g.restore()
    assert not out.rebuilt and out.restoration == "not-needed" and out.snapshot_removed
    assert out.evaluator_image_ref is None and d.image_ids[TASK_IMAGE] == ORIGINAL


def test_guard_restores_even_when_the_tag_vanished():
    d = docker_with_original()
    g = ImageGuard(d, TASK_IMAGE, "run1")
    g.snapshot()
    d.images.discard(TASK_IMAGE)  # evaluator removed the tag and its build failed
    d.image_ids.pop(TASK_IMAGE)
    out = g.restore()
    assert out.restoration == "restored" and out.evaluator_image_id is None
    assert d.image_ids[TASK_IMAGE] == ORIGINAL and d.deleted_ids == []


def _rebuilt_guard(d):
    g = ImageGuard(d, TASK_IMAGE, "run1")
    g.snapshot()
    d.external_rebuild(TASK_IMAGE, REBUILT)
    return g, g.restore()


def test_default_release_removes_only_the_image_this_evaluation_created():
    d = docker_with_original()
    g, out = _rebuilt_guard(d)
    g.release_evaluator_image(out, keep=False)
    assert out.retention == "removed" and out.evaluator_image_ref is None
    assert d.deleted_ids == [REBUILT]  # the evaluator-built image, and nothing else
    assert ORIGINAL not in d.deleted_ids and d.image_ids[TASK_IMAGE] == ORIGINAL
    assert refs(d, EVALUATOR_IMAGE_NAMESPACE) == []  # no multi-GB reference accumulates


def test_keep_release_retains_the_image_under_an_ahl_reference():
    d = docker_with_original()
    g, out = _rebuilt_guard(d)
    g.release_evaluator_image(out, keep=True)
    assert out.retention == "kept" and out.evaluator_image_ref.startswith(
        EVALUATOR_IMAGE_NAMESPACE + "/"
    )
    assert d.image_ids[out.evaluator_image_ref] == REBUILT and d.deleted_ids == []


def test_an_evaluator_image_that_already_existed_is_never_removed():
    d = docker_with_original()
    d.known_ids.add(REBUILT)  # e.g. an older dangling image that happens to be identical
    g, out = _rebuilt_guard(d)
    g.release_evaluator_image(out, keep=False)
    assert out.retention == "skipped-preexisting" and d.deleted_ids == []
    assert REBUILT in d.list_image_ids()


def test_unknown_pre_existing_images_means_nothing_is_removed():
    class CannotList(FakeDocker):
        def list_image_ids(self):
            raise DockerError("cannot list")

    d = CannotList(images={TASK_IMAGE})
    d.image_ids[TASK_IMAGE] = ORIGINAL
    g, out = _rebuilt_guard(d)
    g.release_evaluator_image(out, keep=False)
    assert out.retention == "skipped-unsafe" and d.deleted_ids == []


def test_a_refused_removal_is_recorded_not_forced():
    class Refuses(FakeDocker):
        def remove_image(self, image_id):
            raise DockerError("image is being used")

    d = Refuses(images={TASK_IMAGE})
    d.image_ids[TASK_IMAGE] = ORIGINAL
    g, out = _rebuilt_guard(d)
    g.release_evaluator_image(out, keep=False)
    assert out.retention == "remove-failed" and any(
        "could not be removed" in w for w in out.warnings
    )


def test_no_release_is_needed_when_the_evaluator_did_not_rebuild():
    d = docker_with_original()
    g = ImageGuard(d, TASK_IMAGE, "run1")
    g.snapshot()
    out = g.restore()
    g.release_evaluator_image(out, keep=False)
    assert out.retention == "not-applicable" and d.deleted_ids == []


def test_restoration_failure_is_prominent_and_keeps_the_safe_handle():
    class CannotRestore(FakeDocker):
        def tag_image(self, source, target):
            if target == TASK_IMAGE:
                raise DockerError("daemon refused")
            super().tag_image(source, target)

    d = CannotRestore(images={TASK_IMAGE})
    d.image_ids[TASK_IMAGE] = ORIGINAL
    g = ImageGuard(d, TASK_IMAGE, "run1")
    g.snapshot()
    d.external_rebuild(TASK_IMAGE, REBUILT)
    out = g.restore()
    assert out.restoration == "failed" and not out.snapshot_removed
    assert out.warnings[0].startswith("TASK IMAGE TAG NOT RESTORED")
    assert f"docker tag {ORIGINAL} {TASK_IMAGE}" in out.warnings[0]  # tells the user how to fix
    assert d.image_ids[out.snapshot_ref] == ORIGINAL  # original still safe under the snapshot
    g.release_evaluator_image(out, keep=False)
    assert out.retention == "retained-for-recovery"  # recoverability first: no clean-up
    assert d.deleted_ids == [] and REBUILT in d.list_image_ids()


def test_unverified_restoration_is_reported_as_failed():
    class WrongTag(FakeDocker):
        def tag_image(self, source, target):
            super().tag_image(source, target)
            if target == TASK_IMAGE:
                self.image_ids[TASK_IMAGE] = REBUILT  # tag does not stick

    d = WrongTag(images={TASK_IMAGE})
    d.image_ids[TASK_IMAGE] = ORIGINAL
    g = ImageGuard(d, TASK_IMAGE, "run1")
    g.snapshot()
    d.external_rebuild(TASK_IMAGE, REBUILT)
    assert g.restore().restoration == "failed"


def test_snapshot_failure_means_the_evaluator_is_never_started(run_env):
    class NoSnapshot(FakeDocker):
        def tag_image(self, source, target):
            raise DockerError("cannot tag")

    runner = FakeEvaluatorRunner()
    d = NoSnapshot(images={TASK_IMAGE})
    with pytest.raises(EvaluationError, match="cannot protect the task image tag"):
        evaluate(run_env, d, runner)
    assert not [c for c in runner.calls if "eval" in c["argv"]]  # unprotected tag: do not run
    assert not (run_env["run_dir"] / "evaluation.json").exists()


# ---- through the real evaluation flow ----
def test_original_tag_is_preserved_after_a_successful_evaluation(run_env):
    d = docker_with_original()
    runner = FakeEvaluatorRunner(docker=d, rebuild_to=REBUILT)
    _, doc = evaluate(run_env, d, runner)
    assert doc.state is EvaluationState.RESOLVED
    assert d.image_ids[TASK_IMAGE] == ORIGINAL and ORIGINAL not in d.deleted_ids
    assert refs(d, SNAPSHOT_NAMESPACE) == []


def test_original_tag_is_preserved_after_an_evaluator_failure(run_env):
    d = docker_with_original()
    runner = FakeEvaluatorRunner(docker=d, rebuild_to=REBUILT, report=None, returncode=1)
    _, doc = evaluate(run_env, d, runner)
    assert doc.state is EvaluationState.INFRA_FAILURE and doc.resolved is None
    assert d.image_ids[TASK_IMAGE] == ORIGINAL and refs(d, SNAPSHOT_NAMESPACE) == []
    assert doc.image.restoration == "restored"


def test_original_tag_is_preserved_when_the_evaluator_cannot_even_start(run_env):
    d = docker_with_original()
    _, doc = evaluate(run_env, d, FakeEvaluatorRunner(start_error="exec format error", docker=d))
    assert doc.state is EvaluationState.INFRA_FAILURE and d.image_ids[TASK_IMAGE] == ORIGINAL


def test_original_tag_is_preserved_after_an_interruption(run_env):
    d = docker_with_original()
    runner = FakeEvaluatorRunner(docker=d, rebuild_to=REBUILT, interrupt=True)
    with pytest.raises(KeyboardInterrupt):
        evaluate(run_env, d, runner)
    assert d.image_ids[TASK_IMAGE] == ORIGINAL and ORIGINAL not in d.deleted_ids
    assert refs(d, SNAPSHOT_NAMESPACE) == []
    doc = json.loads((run_env["run_dir"] / "evaluation.json").read_text())
    assert doc["interrupted"] is True and doc["resolved"] is None  # evidence kept, no verdict
    assert doc["image"]["restoration"] == "restored"


def test_separate_image_identities_are_recorded(run_env):
    d = docker_with_original()
    _, doc = evaluate(run_env, d, FakeEvaluatorRunner(docker=d, rebuild_to=REBUILT))
    img = doc.image
    assert img.agent_image_id == FakeDocker.DEFAULT_ID  # what the agent ran in (run.json)
    assert img.image_id_before == ORIGINAL  # the tag when evaluation started
    assert img.evaluator_image_id == REBUILT  # what the tests ran in
    assert img.final_image_id == ORIGINAL  # the tag afterwards
    assert img.rebuilt and img.restoration == "restored" and img.snapshot_removed
    assert img.retention == "removed" and img.evaluator_image_ref is None  # default: not kept
    assert img.snapshot_ref and not img.keep_requested
    on_disk = json.loads((run_env["run_dir"] / "evaluation.json").read_text())["image"]
    assert {on_disk["image_id_before"], on_disk["evaluator_image_id"]} == {ORIGINAL, REBUILT}


def test_a_rebuild_is_never_hidden(run_env):
    d = docker_with_original()
    _, doc = evaluate(run_env, d, FakeEvaluatorRunner(docker=d, rebuild_to=REBUILT))
    assert any("REBUILT" in w and REBUILT in w for w in doc.warnings)
    assert any("already differed from the agent's image" in w for w in doc.warnings)


def test_restoration_failure_is_preserved_in_the_evaluation_record(run_env):
    class CannotRestore(FakeDocker):
        def tag_image(self, source, target):
            if target == TASK_IMAGE:
                raise DockerError("daemon refused")
            super().tag_image(source, target)

    d = CannotRestore(images={TASK_IMAGE})
    d.image_ids[TASK_IMAGE] = ORIGINAL
    _, doc = evaluate(run_env, d, FakeEvaluatorRunner(docker=d, rebuild_to=REBUILT))
    assert doc.image.restoration == "failed" and not doc.image.snapshot_removed
    assert any("NOT RESTORED" in w for w in doc.warnings)
    assert doc.state is EvaluationState.RESOLVED  # the benchmark verdict itself is unaffected
    assert doc.image.retention == "retained-for-recovery"
    assert d.deleted_ids == [] and REBUILT in d.list_image_ids()  # evidence kept


def test_cli_exit_code_flags_a_failed_restoration(run_env, monkeypatch):
    from typer.testing import CliRunner

    from agent_harness_lab.cli import common
    from agent_harness_lab.cli import evaluate as evaluate_cli
    from agent_harness_lab.cli.app import app
    from tests.benchmark_fixtures import FakeGit

    class CannotRestore(FakeDocker):
        def tag_image(self, source, target):
            if target == TASK_IMAGE:
                raise DockerError("daemon refused")
            super().tag_image(source, target)

    d = CannotRestore(images={TASK_IMAGE})
    d.image_ids[TASK_IMAGE] = ORIGINAL
    monkeypatch.setattr(common, "docker_client", lambda: d)
    monkeypatch.setattr(
        common, "command_runner", lambda: FakeEvaluatorRunner(docker=d, rebuild_to=REBUILT)
    )
    monkeypatch.setattr(common, "environ", lambda: {})
    monkeypatch.setattr(evaluate_cli, "_git", lambda: FakeGit())
    monkeypatch.setenv("AHL_SWEBENCH_TASKS_REPO", str(run_env["repo"]))
    r = CliRunner().invoke(
        app,
        ["evaluate", str(run_env["run_dir"]), "--swebench-python", str(run_env["python"])],
        env={"COLUMNS": "200"},
    )
    assert r.exit_code == 1 and "TASK IMAGE TAG NOT RESTORED" in r.output


def test_two_evaluations_of_the_same_task_each_record_their_own_evaluator_image(
    tmp_path, task_repo
):
    """Each official evaluation rebuilds, so the evaluator image identities can differ; both are
    recorded so a pair of runs (e.g. Pi vs OpenCode) can be checked for a shared identity."""
    d = docker_with_original()
    ids = []
    for n, rebuilt in enumerate((REBUILT, "sha256:" + "c" * 64)):
        run_dir = make_ahl_run(tmp_path / f"r{n}", tmp_path / "raw")
        ctx = make_ctx(
            run_dir,
            task_repo,
            make_python(tmp_path / f"r{n}"),
            FakeEvaluatorRunner(docker=d, rebuild_to=rebuilt),
            docker=d,
        )
        _, doc = run_evaluation(ADAPTER, ctx)
        ids.append(doc.image.evaluator_image_id)
        assert d.image_ids[TASK_IMAGE] == ORIGINAL  # restored after EACH evaluation
    assert ids[0] != ids[1]  # visible, not hidden: the pair did NOT share an evaluator image


def test_official_verdict_parsing_is_unchanged_by_the_image_handling(run_env):
    d = docker_with_original()
    _, doc = evaluate(
        run_env, d, FakeEvaluatorRunner(docker=d, rebuild_to=REBUILT, bucket="unresolved_ids")
    )
    assert doc.state is EvaluationState.UNRESOLVED and doc.resolved is False
