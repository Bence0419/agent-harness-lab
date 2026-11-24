import json
from datetime import UTC, datetime, timedelta

import pytest

from agent_harness_lab.artifacts.hashing import sha256_file, sha256_text
from agent_harness_lab.execution.executor import RunRequest, allocate_run_dir, execute_run
from agent_harness_lab.execution.models import (
    DockerError,
    ExecutionDisabled,
    ExecutionState,
    ImageMissing,
)
from agent_harness_lab.execution.network import SQUID_IMAGE
from agent_harness_lab.harnesses.base import GeneratedFile
from agent_harness_lab.runplan.models import NetworkMode
from tests.fake_docker import PATCH_TEXT, FakeDocker
from tests.mock_harness import MOCK_CONTAINER_PATH, mock_plan, mock_provisioner, write_mock_harness

PROMPT = "Fix the mock issue."
SECRET = "sk-test-SECRET-value-123"


class Clock:
    """Deterministic UTC clock advancing 1 s per call."""

    def __init__(self):
        self.t = datetime(2026, 11, 24, 10, 0, 0, tzinfo=UTC)

    def __call__(self):
        self.t += timedelta(seconds=1)
        return self.t


def run(tmp_path, docker=None, plan=None, env=None, **kw):
    plan = plan or mock_plan()
    host = write_mock_harness(tmp_path)
    request = RunRequest(
        plan=plan, prompt=PROMPT, provisioner=mock_provisioner(host), raw_dir=tmp_path / "raw"
    )
    docker = docker or FakeDocker()
    doc = execute_run(
        request, docker=docker, now=Clock(), sleep=lambda s: None, environ=env or {}, **kw
    )
    return doc, docker, tmp_path / "raw" / "swebench" / plan.task.instance_id / doc.run_id


# ---- success and layout ----
def test_successful_run_creates_all_artifacts(tmp_path):
    doc, docker, d = run(tmp_path)
    assert doc.state is ExecutionState.COMPLETED and doc.exit_code == 0
    for name in ("run.json", "metadata.json", "prompt.txt", "trace.jsonl", "model.patch",
                 "git-status.txt", "mock.stderr.log"):  # fmt: skip
        assert (d / name).is_file(), name
    assert (d / "prompt.txt").read_text() == PROMPT
    assert (d / "model.patch").read_text() == PATCH_TEXT
    assert d.parent.parent.name == "swebench" and d.parent.name == "mock__task-1"
    assert doc.run_id.startswith("mock__task-1-mock-20261124T100001Z")


def test_operation_order_matches_the_conceptual_flow(tmp_path):
    _, docker, _ = run(tmp_path)
    ops = docker.ops()
    assert ops.index("image_inspect") < ops.index("create_container") < ops.index("start")
    assert ops.index("start") < ops.index("exec") < ops.index("remove")
    assert ops[-1] == "remove"  # cleanup is last


def test_harness_exec_uses_workdir_and_substituted_prompt(tmp_path):
    _, docker, _ = run(tmp_path)
    harness = next(e for e in docker.exec_log if e["argv"][0] == MOCK_CONTAINER_PATH)
    assert harness["argv"] == [MOCK_CONTAINER_PATH, "--behavior", "edit", PROMPT]
    assert harness["workdir"] == "/testbed"


def test_run_json_contents(tmp_path):
    doc, _, d = run(tmp_path)
    data = json.loads((d / "run.json").read_text())
    assert data["schema_version"] == "ahl-run-v1"
    assert data["run_id"] == doc.run_id and data["state"] == "completed"
    assert data["plan"]["task"]["base_commit"] == "a" * 40
    assert data["plan"]["ahl"] == {"commit": "c" * 40, "dirty": False, "version": None}
    assert data["plan"]["task"]["task_repository_commit"] == "b" * 40
    assert data["image"]["reference"] == "ahl-test-synthetic:1"
    assert data["image"]["image_id"] == "sha256:" + "1" * 64  # resolved id, not just the tag
    assert data["container_id"].startswith("container-id-ahl-agent-")
    assert data["started_at"].endswith("Z") and data["finished_at"].endswith("Z")
    assert data["duration_seconds"] == 1.0 and data["exit_code"] == 0
    assert data["cleanup"] == {"status": "ok", "errors": []}
    assert data["network"]["mode"] == "unrestricted"
    assert data["plan"]["timeout_seconds"] == 3600


def test_hashes_are_sha256_of_the_finalized_files(tmp_path):
    doc, _, d = run(tmp_path)
    h = json.loads((d / "run.json").read_text())["hashes"]
    assert h["prompt_sha256"] == sha256_text(PROMPT) == sha256_file(d / "prompt.txt")
    assert h["patch_sha256"] == sha256_file(d / "model.patch")
    assert h["trace_sha256"] == sha256_file(d / "trace.jsonl")
    assert h["artifacts"]["git-status.txt"] == sha256_file(d / "git-status.txt")
    assert "run.json" not in h["artifacts"]  # a file cannot contain its own hash


def test_provisioning_is_recorded_structurally(tmp_path):
    _, docker, d = run(tmp_path)
    prov = json.loads((d / "run.json").read_text())["provisioning"]
    assert prov[0]["kind"] == "bind-mount" and prov[0]["destination"] == MOCK_CONTAINER_PATH
    assert prov[0]["read_only"] is True and prov[0]["source"].endswith("mock-harness")
    created = next(c for c in docker.calls if c[0] == "create_container")
    assert created[1][0].startswith("ahl-agent-")


def test_container_spec_labels_env_and_mounts(tmp_path):
    docker = FakeDocker()
    captured = {}
    orig = docker.create_container
    docker.create_container = lambda spec: (captured.setdefault("spec", spec), orig(spec))[1]
    doc, _, _ = run(tmp_path, docker)
    spec = captured["spec"]
    assert spec.labels == {"ahl.managed": "true", "ahl.run-id": doc.run_id}
    assert spec.env == {"MOCK_ENV": "1"} and spec.network is None  # unrestricted: default bridge
    assert [m.destination for m in spec.mounts] == [MOCK_CONTAINER_PATH]
    assert spec.name == "ahl-agent-" + doc.run_id


def test_generated_config_is_staged_hashed_and_copied_in(tmp_path):
    cfg = GeneratedFile(container_path="/etc/mock/conf.json", content='{"a": 1}\n', description="c")
    doc, docker, d = run(tmp_path, plan=mock_plan(config_files=[cfg]))
    staged = d / "generated" / "00-conf.json"
    assert staged.read_text() == '{"a": 1}\n'
    assert doc.hashes.generated_config_sha256 == {"/etc/mock/conf.json": sha256_text('{"a": 1}\n')}
    prov = [p for p in doc.provisioning if p.kind == "copy-in"]
    assert prov[0].destination == "/etc/mock/conf.json" and prov[0].sha256 == sha256_text(
        '{"a": 1}\n'
    )
    assert ["mkdir", "-p", "/etc/mock"] in [e["argv"] for e in docker.exec_log]
    assert docker.copies == [(str(staged), docker.copies[0][1], "/etc/mock/conf.json")]


# ---- harness outcomes ----
def test_nonzero_exit_is_harness_failed_and_evidence_is_kept(tmp_path):
    doc, _, d = run(tmp_path, FakeDocker(harness_rc=3), plan=mock_plan(behavior="fail"))
    assert doc.state is ExecutionState.HARNESS_FAILED and doc.exit_code == 3
    assert (d / "model.patch").read_text() == PATCH_TEXT  # patch still captured
    assert doc.failed_stage is None  # not an infrastructure failure


def test_timeout_with_integer_applies_it_and_is_recorded(tmp_path):
    doc, docker, d = run(
        tmp_path, FakeDocker(harness_rc=-15, harness_timed_out=True), plan=mock_plan(timeout=7)
    )
    harness = next(e for e in docker.exec_log if e["argv"][0] == MOCK_CONTAINER_PATH)
    assert harness["timeout"] == 7
    assert doc.state is ExecutionState.TIMED_OUT
    assert (d / "model.patch").exists() and doc.cleanup.status == "ok"
    assert json.loads((d / "run.json").read_text())["plan"]["timeout_seconds"] == 7


def test_timeout_none_means_no_timeout_wrapper_at_all(tmp_path):
    doc, docker, _ = run(tmp_path, plan=mock_plan(timeout=None))
    harness = next(e for e in docker.exec_log if e["argv"][0] == MOCK_CONTAINER_PATH)
    assert harness["timeout"] is None  # never silently added
    assert doc.plan.timeout_seconds is None and doc.state is ExecutionState.COMPLETED


def test_empty_patch_is_a_valid_completed_run(tmp_path):
    doc, _, d = run(tmp_path, FakeDocker(patch_text=""), plan=mock_plan(behavior="noop"))
    assert doc.state is ExecutionState.COMPLETED
    assert (d / "model.patch").read_text() == ""
    assert doc.hashes.patch_sha256 == sha256_text("")  # genuine empty patch, hashed


# ---- failures ----
def test_missing_image_fails_clearly_without_pull_build_or_container(tmp_path):
    doc, docker, d = run(tmp_path, FakeDocker(images=set()))
    assert doc.state is ExecutionState.FAILED and doc.failed_stage == "image"
    assert "not found locally" in doc.error
    assert "create_container" not in docker.ops() and "remove" not in docker.ops()
    assert doc.cleanup.status == "not-needed"
    assert json.loads((d / "run.json").read_text())["state"] == "failed"  # valid record exists


def test_container_creation_failure(tmp_path):
    doc, docker, _ = run(
        tmp_path, FakeDocker(fail_on={"create_container": DockerError("no space")})
    )
    assert (doc.state, doc.failed_stage) == (ExecutionState.FAILED, "container")
    assert "no space" in doc.error and "exec" not in docker.ops()


def test_workspace_failure_still_cleans_up(tmp_path):
    doc, docker, d = run(tmp_path, FakeDocker(head="f" * 40))
    assert (doc.state, doc.failed_stage) == (ExecutionState.FAILED, "workspace")
    assert docker.ops()[-1] == "remove" and doc.cleanup.status == "ok"
    assert not any(
        e["argv"][0] == MOCK_CONTAINER_PATH for e in docker.exec_log
    )  # harness never ran


def test_patch_capture_failure_marks_the_run_failed(tmp_path):
    docker = FakeDocker()
    real_exec = docker.exec

    def exec_(container, argv, **kw):
        if "diff" in argv:
            from agent_harness_lab.execution.models import ExecOutput

            return ExecOutput(returncode=128)
        return real_exec(container, argv, **kw)

    docker.exec = exec_
    doc, _, _ = run(tmp_path, docker)
    assert (doc.state, doc.failed_stage) == (ExecutionState.FAILED, "patch")


def test_cleanup_failure_is_recorded_without_erasing_the_original_failure(tmp_path):
    docker = FakeDocker(head="f" * 40, fail_on={"remove": DockerError("daemon went away")})
    doc, _, d = run(tmp_path, docker)
    assert (doc.state, doc.failed_stage) == (ExecutionState.FAILED, "workspace")  # original kept
    assert "unexpected repository HEAD" in doc.error
    assert doc.cleanup.status == "failed" and "daemon went away" in doc.cleanup.errors[0]
    on_disk = json.loads((d / "run.json").read_text())
    assert on_disk["failed_stage"] == "workspace" and on_disk["cleanup"]["status"] == "failed"


def test_cleanup_failure_after_success_is_recorded_and_state_stays_completed(tmp_path):
    doc, _, _ = run(tmp_path, FakeDocker(fail_on={"remove": DockerError("busy")}))
    assert doc.state is ExecutionState.COMPLETED and doc.cleanup.status == "failed"


def test_unexpected_exception_leaves_a_valid_run_json_and_still_cleans_up(tmp_path):
    docker = FakeDocker(fail_on={"start": RuntimeError("unexpected bug")})
    doc, _, d = run(tmp_path, docker)
    assert (doc.state, doc.failed_stage) == (ExecutionState.FAILED, "internal")
    assert "unexpected bug" in doc.error and docker.ops()[-1] == "remove"
    assert json.loads((d / "run.json").read_text())["failed_stage"] == "internal"


def test_keyboard_interrupt_records_state_cleans_up_and_propagates(tmp_path):
    docker = FakeDocker(fail_on={"start": KeyboardInterrupt()})
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, docker)
    (d,) = list((tmp_path / "raw" / "swebench" / "mock__task-1").iterdir())
    data = json.loads((d / "run.json").read_text())
    assert data["failed_stage"] == "interrupted" and data["cleanup"]["status"] == "ok"


# ---- restricted network end to end (fake) ----
def test_restricted_network_run_collects_proxy_log_and_removes_everything(tmp_path):
    plan = mock_plan(network=NetworkMode.RESTRICTED_PROVIDER, allowed_hosts=["llm.example.org"])
    doc, docker, d = run(tmp_path, plan=plan)
    assert doc.state is ExecutionState.COMPLETED
    assert (d / "proxy.log").read_text() == "proxy log line\n"
    assert (d / "generated" / "squid.conf").read_text().count("llm.example.org") == 1
    assert docker.networks == set() and docker.proxies == set()  # nothing left behind
    assert doc.network.squid_image == SQUID_IMAGE and len(doc.network.squid_policy_sha256) == 64
    created = next(c for c in docker.calls if c[0] == "create_container")
    spec_net = next(o for o in docker.calls if o[0] == "network_create")[1][0]
    assert created[1][0].startswith("ahl-agent-") and spec_net.startswith("ahl-net-")
    ops = docker.ops()
    assert ops.index("logs") < ops.index("remove") < ops.index("network_remove")


def test_restricted_run_failing_midway_still_removes_network_and_proxy(tmp_path):
    plan = mock_plan(network=NetworkMode.RESTRICTED_PROVIDER, allowed_hosts=["llm.example.org"])
    doc, docker, _ = run(tmp_path, FakeDocker(head="f" * 40), plan=plan)
    assert doc.failed_stage == "workspace"
    assert docker.networks == set() and docker.proxies == set() and doc.cleanup.status == "ok"


def test_squid_image_missing_fails_the_network_stage(tmp_path):
    plan = mock_plan(network=NetworkMode.RESTRICTED_PROVIDER, allowed_hosts=["h.example.org"])
    doc, docker, _ = run(tmp_path, FakeDocker(images={"ahl-test-synthetic:1"}), plan=plan)
    assert doc.failed_stage == "network" and "create_container" not in docker.ops()


# ---- run directory handling ----
def test_run_directories_are_never_reused(tmp_path):
    raw = tmp_path / "raw"
    now = datetime(2026, 11, 24, 10, 0, 0, tzinfo=UTC)
    a_id, a = allocate_run_dir(raw, "t__t-1", "pi", now)
    b_id, b = allocate_run_dir(raw, "t__t-1", "pi", now)  # same second
    assert a != b and b_id == a_id + "-2" and a.is_dir() and b.is_dir()
    assert a_id == "t__t-1-pi-20261124T100000Z"


def test_existing_run_directory_content_is_never_overwritten(tmp_path):
    raw = tmp_path / "raw"
    now = datetime(2026, 11, 24, 10, 0, 0, tzinfo=UTC)
    _, first = allocate_run_dir(raw, "t__t-1", "pi", now)
    (first / "run.json").write_text("precious")
    allocate_run_dir(raw, "t__t-1", "pi", now)
    assert (first / "run.json").read_text() == "precious"


# ---- secrets ----
def test_missing_required_secret_fails_closed_before_any_docker_call(tmp_path):
    doc, docker, _ = run(tmp_path, plan=mock_plan(secret_env=["MOCK_API_KEY"]), env={})
    assert (doc.state, doc.failed_stage) == (ExecutionState.FAILED, "environment")
    assert "MOCK_API_KEY" in doc.error and docker.calls == []


def test_secret_values_never_reach_run_json_metadata_or_logs(tmp_path):
    docker = FakeDocker(fail_on={"start": DockerError(f"start failed with key {SECRET}")})
    doc, _, d = run(tmp_path, docker, plan=mock_plan(secret_env=["MOCK_API_KEY"]),
                    env={"MOCK_API_KEY": SECRET})  # fmt: skip
    assert doc.failed_stage == "container"
    for path in d.rglob("*"):
        if path.is_file():
            assert SECRET not in path.read_text(), path.name
    assert "***" in doc.error
    assert doc.plan.required_secret_env == ["MOCK_API_KEY"]  # the NAME may be recorded


def test_secret_is_forwarded_by_name_to_the_container_spec(tmp_path):
    docker = FakeDocker()
    seen = {}
    orig = docker.create_container
    docker.create_container = lambda spec: (seen.update(spec=spec), orig(spec))[1]
    run(tmp_path, docker, plan=mock_plan(secret_env=["MOCK_API_KEY"]), env={"MOCK_API_KEY": SECRET})
    assert seen["spec"].pass_env == ["MOCK_API_KEY"]
    assert SECRET not in seen["spec"].model_dump_json()


# ---- guards ----
@pytest.mark.parametrize("name", ["pi", "opencode"])
def test_real_harness_execution_is_disabled_and_touches_nothing(tmp_path, name):
    docker = FakeDocker()
    request = RunRequest(
        plan=mock_plan(harness_name=name),
        prompt=PROMPT,
        provisioner=mock_provisioner(write_mock_harness(tmp_path)),
        raw_dir=tmp_path / "raw",
    )
    with pytest.raises(ExecutionDisabled, match="disabled.*AHL_ENABLE_REAL_EXECUTION"):
        execute_run(request, docker=docker)
    assert docker.calls == [] and not (tmp_path / "raw").exists()


def test_prompt_mismatch_with_the_plan_fails_the_prompt_stage(tmp_path):
    host = write_mock_harness(tmp_path)
    request = RunRequest(
        plan=mock_plan(prompt="the planned prompt"), prompt="a different prompt",
        provisioner=mock_provisioner(host), raw_dir=tmp_path / "raw",
    )  # fmt: skip
    doc = execute_run(request, docker=FakeDocker(), now=Clock(), sleep=lambda s: None)
    assert (doc.state, doc.failed_stage) == (ExecutionState.FAILED, "prompt")


def test_execution_states_exist():
    assert {s.value for s in ExecutionState} >= {
        "prepared", "running", "completed", "failed", "timed_out",
    }  # fmt: skip


def test_image_missing_error_type():
    assert issubclass(ImageMissing, DockerError)


# ---- timeout cleanup: kill only this run's container ----
def timed_out_run(tmp_path, docker=None, timeout=5):
    docker = docker or FakeDocker(harness_rc=-15, harness_timed_out=True)
    return run(tmp_path, docker, plan=mock_plan(timeout=timeout))


def test_timeout_kills_the_managed_container_then_restores_it_for_evidence(tmp_path):
    doc, docker, d = timed_out_run(tmp_path)
    kills = [args for op, args in docker.calls if op == "kill"]
    assert kills == [(f"ahl-agent-{doc.run_id}",)]  # exactly one kill, of exactly this container
    ops = docker.ops()
    harness_at = max(i for i, o in enumerate(ops) if o == "exec" and i < ops.index("kill"))
    assert harness_at < ops.index("kill") < len(ops) - 1
    assert ops[ops.index("kill") + 1] == "start"  # quiescent container for patch capture
    assert doc.timeout_action == "killed"
    assert json.loads((d / "run.json").read_text())["timeout_action"] == "killed"


def test_timed_out_is_recorded_on_disk_before_the_kill(tmp_path):
    docker = FakeDocker(harness_rc=-15, harness_timed_out=True)
    seen = {}

    def at_kill():
        (path,) = list((tmp_path / "raw" / "swebench" / "mock__task-1").glob("*/run.json"))
        seen["state"] = json.loads(path.read_text())["state"]

    docker.on_kill = at_kill
    timed_out_run(tmp_path, docker)
    assert seen == {"state": "timed_out"}


def test_unrelated_containers_are_never_touched(tmp_path):
    docker = FakeDocker(harness_rc=-15, harness_timed_out=True)
    bystander = "unrelated-service"
    docker.containers[bystander] = None  # an existing container on the same daemon
    doc, _, _ = timed_out_run(tmp_path, docker)
    touched = {
        a
        for op, args in docker.calls
        if op in {"kill", "start", "remove", "exec"}
        for a in args[:1]
    }
    assert bystander not in touched and bystander in docker.containers
    named = [args for op, args in docker.calls if op in {"kill", "start", "remove"}]
    assert all(bystander not in str(args) for args in named)


def test_timed_out_remains_the_final_state_and_patch_is_still_captured(tmp_path):
    doc, docker, d = timed_out_run(tmp_path)
    assert doc.state is ExecutionState.TIMED_OUT and doc.failed_stage is None
    assert (d / "model.patch").read_text() == PATCH_TEXT
    assert json.loads((d / "run.json").read_text())["state"] == "timed_out"


def test_cleanup_still_runs_after_a_timeout_kill(tmp_path):
    doc, docker, _ = timed_out_run(tmp_path)
    assert docker.ops()[-1] == "remove" and doc.cleanup.status == "ok"
    assert docker.ops().index("kill") < docker.ops().index("remove")


def test_kill_failure_is_recorded_without_replacing_the_timeout(tmp_path):
    docker = FakeDocker(
        harness_rc=-15, harness_timed_out=True, fail_on={"kill": DockerError("kill refused")}
    )
    doc, _, d = timed_out_run(tmp_path, docker)
    assert doc.state is ExecutionState.TIMED_OUT and doc.failed_stage is None
    assert doc.timeout_action == "kill-failed"
    assert any("kill after timeout failed: kill refused" in w for w in doc.warnings)
    assert (d / "model.patch").exists() and docker.ops()[-1] == "remove"  # evidence + cleanup go on
    assert docker.ops().count("start") == 1  # no restart attempted after a failed kill


def test_restart_failure_after_the_kill_keeps_the_timeout_and_still_cleans_up(tmp_path):
    class FlakyStart(FakeDocker):
        starts = 0

        def start(self, name):
            FlakyStart.starts += 1
            if FlakyStart.starts == 2:  # the restart after the kill
                raise DockerError("cannot restart")
            super().start(name)

    FlakyStart.starts = 0
    doc, docker, _ = timed_out_run(tmp_path, FlakyStart(harness_rc=-15, harness_timed_out=True))
    assert doc.state is ExecutionState.TIMED_OUT and doc.timeout_action == "killed"
    assert any("restart for evidence capture failed" in w for w in doc.warnings)
    assert docker.ops()[-1] == "remove" and doc.cleanup.status == "ok"


def test_cleanup_failure_after_a_timeout_does_not_replace_it(tmp_path):
    docker = FakeDocker(
        harness_rc=-15, harness_timed_out=True, fail_on={"remove": DockerError("busy")}
    )
    doc, _, _ = timed_out_run(tmp_path, docker)
    assert doc.state is ExecutionState.TIMED_OUT and doc.cleanup.status == "failed"


def test_kill_failure_message_cannot_leak_a_secret(tmp_path):
    docker = FakeDocker(
        harness_rc=-15, harness_timed_out=True, fail_on={"kill": DockerError(f"token {SECRET}")}
    )
    doc, _, d = run(
        tmp_path, docker, plan=mock_plan(timeout=5, secret_env=["MOCK_API_KEY"]),
        env={"MOCK_API_KEY": SECRET},
    )  # fmt: skip
    assert SECRET not in (d / "run.json").read_text() and doc.state is ExecutionState.TIMED_OUT


def test_timeout_none_never_kills_anything(tmp_path):
    doc, docker, _ = run(tmp_path, plan=mock_plan(timeout=None))
    assert "kill" not in docker.ops() and doc.timeout_action is None
    assert doc.state is ExecutionState.COMPLETED


def test_success_and_harness_failure_never_trigger_a_kill_and_are_unchanged(tmp_path):
    ok, docker_ok, _ = run(tmp_path)
    (tmp_path / "second").mkdir()
    failed, docker_failed, _ = run(tmp_path / "second", FakeDocker(harness_rc=3), plan=mock_plan())
    assert ok.state is ExecutionState.COMPLETED and failed.state is ExecutionState.HARNESS_FAILED
    assert "kill" not in docker_ok.ops() + docker_failed.ops()
    assert ok.timeout_action is None and failed.timeout_action is None
    assert docker_ok.ops().count("start") == 1  # no restart outside the timeout path
