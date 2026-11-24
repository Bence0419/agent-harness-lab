"""Opt-in integration tests against a REAL Docker daemon (``pytest -m docker``).

No LLM, no Pi/OpenCode, no network access to anything external, no secrets. They use:
  * a tiny synthetic task image committed from a locally available image (no pull, no build),
  * the deterministic mock harness,
  * for the network test, local nginx as an "allowed" origin on a private Docker network.
Every resource is labelled ``ahl.managed=true`` (or removed by the fixtures) and verified gone.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from datetime import UTC, datetime

import pytest

from agent_harness_lab.execution.docker import DockerCli
from agent_harness_lab.execution.executor import RunRequest, execute_run
from agent_harness_lab.execution.models import ExecutionState
from agent_harness_lab.execution.network import plan_network, start_network
from agent_harness_lab.runplan.models import NetworkMode, NetworkPlan
from tests.mock_harness import EDIT_LINE, mock_plan, mock_provisioner, write_mock_harness

pytestmark = pytest.mark.docker

BASE_IMAGE = "python:3.14-trixie"  # provides git + sh and is already present locally
PROMPT = "Fix the mock issue."


def sh(*argv, check=True) -> str:
    r = subprocess.run(list(argv), capture_output=True, text=True)
    if check and r.returncode != 0:
        raise AssertionError(f"{argv} failed: {r.stderr}")
    return r.stdout.strip()


def have_image(ref: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", ref], capture_output=True).returncode == 0


def managed(kind: str) -> list[str]:
    cmd = {"container": ["docker", "ps", "-aq"], "network": ["docker", "network", "ls", "-q"]}[kind]
    return sh(*cmd, "--filter", "label=ahl.managed=true").split()


@pytest.fixture(scope="module", autouse=True)
def docker_ready():
    if (
        shutil.which("docker") is None
        or subprocess.run(["docker", "info"], capture_output=True).returncode != 0
    ):
        pytest.skip("Docker daemon not available")
    if not have_image(BASE_IMAGE):
        pytest.skip(f"{BASE_IMAGE} not present locally (this suite never pulls)")


@pytest.fixture(scope="module")
def synthetic_task():
    """(image tag, base commit): /testbed is a one-commit git repo containing mock_target.txt."""
    tag = f"ahl-test-synthetic:{uuid.uuid4().hex[:8]}"
    name = f"ahl-test-build-{uuid.uuid4().hex[:8]}"
    sh("docker", "run", "-d", "--name", name, "--entrypoint", "sleep", BASE_IMAGE, "infinity")
    try:
        sh(
            "docker", "exec", name, "sh", "-c",
            "git init -q /testbed && cd /testbed && git config user.email t@example.test "
            "&& git config user.name t && printf 'original\\n' > mock_target.txt "
            "&& git add . && git commit -q -m base",
        )  # fmt: skip
        commit = sh("docker", "exec", name, "git", "-C", "/testbed", "rev-parse", "HEAD")
        sh("docker", "commit", name, tag)
    finally:
        sh("docker", "rm", "-f", name, check=False)
    yield tag, commit
    sh("docker", "rmi", "-f", tag, check=False)


def run_mock(tmp_path, synthetic_task, **plan_kw):
    tag, commit = synthetic_task
    plan = mock_plan(image=tag, base_commit=commit, prompt=PROMPT, **plan_kw)
    request = RunRequest(
        plan=plan,
        prompt=PROMPT,
        provisioner=mock_provisioner(write_mock_harness(tmp_path)),
        raw_dir=tmp_path / "raw",  # dedicated temp location, never results/raw
    )
    before = (len(managed("container")), len(managed("network")))
    doc = execute_run(request, docker=DockerCli())
    d = tmp_path / "raw" / "swebench" / plan.task.instance_id / doc.run_id
    assert (len(managed("container")), len(managed("network"))) == before  # nothing left behind
    return doc, d


def test_mock_run_full_flow_with_real_docker(tmp_path, synthetic_task):
    doc, d = run_mock(tmp_path, synthetic_task)
    assert doc.state is ExecutionState.COMPLETED, doc.error
    assert doc.cleanup.status == "ok" and doc.exit_code == 0

    patch = (d / "model.patch").read_text()
    assert f"+{EDIT_LINE}" in patch and "mock_target.txt" in patch  # deterministic edit captured
    assert (d / "git-status.txt").read_text().strip() == "M mock_target.txt"
    trace = [json.loads(line) for line in (d / "trace.jsonl").read_text().splitlines()]
    assert [e["type"] for e in trace] == ["mock_start", "mock_edit", "mock_end"]
    assert (d / "prompt.txt").read_text() == PROMPT

    run = json.loads((d / "run.json").read_text())
    assert run["schema_version"] == "ahl-run-v1" and run["state"] == "completed"
    assert run["image"]["image_id"].startswith("sha256:")  # resolved id from Docker
    assert run["container_id"] and run["hashes"]["patch_sha256"]
    assert json.loads((d / "metadata.json").read_text())["schema"] == "ahl-metadata-v1"
    blob = (d / "run.json").read_text().lower()
    assert '"api_key":' not in blob and "bearer" not in blob  # no secret-bearing fields


def test_workspace_is_reset_to_the_base_commit_first(tmp_path, synthetic_task):
    doc, d = run_mock(tmp_path, synthetic_task, behavior="noop")
    assert doc.state is ExecutionState.COMPLETED
    assert (d / "model.patch").read_text() == ""  # a genuine empty patch
    assert doc.hashes.patch_sha256 is not None


def test_nonzero_exit_is_harness_failed_with_evidence(tmp_path, synthetic_task):
    doc, d = run_mock(tmp_path, synthetic_task, behavior="fail")
    assert doc.state is ExecutionState.HARNESS_FAILED and doc.exit_code == 3
    assert f"+{EDIT_LINE}" in (d / "model.patch").read_text()
    assert "mock failure" in (d / "mock.stderr.log").read_text()


class _SpyCli(DockerCli):
    """Real Docker; before the container is removed, records whether the agent still runs."""

    survivors: list[str] | None = None

    def remove(self, *names):
        for name in names:
            if name.startswith("ahl-agent-"):
                out = self.exec(
                    name,
                    ["python3", "-c",
                     "import os\nme = str(os.getpid())\nneedle = ('mock-' + 'harness').encode()\n"
                     "for p in os.listdir('/proc'):\n"
                     "  if not p.isdigit() or p == me:\n"
                     "    continue\n"
                     "  try:\n"
                     "    if needle in open('/proc/' + p + '/cmdline', 'rb').read(): print(p)\n"
                     "  except Exception: pass"],
                )  # fmt: skip
                self.survivors = out.stdout.split()
        super().remove(*names)


def test_integer_timeout_kills_only_this_runs_container_and_the_agent(tmp_path, synthetic_task):
    bystander = f"ahl-test-bystander-{uuid.uuid4().hex[:8]}"
    sh("docker", "run", "-d", "--name", bystander, "--entrypoint", "sleep", BASE_IMAGE, "infinity")
    try:
        tag, commit = synthetic_task
        plan = mock_plan(image=tag, base_commit=commit, prompt=PROMPT, behavior="sleep", timeout=3)
        request = RunRequest(
            plan=plan, prompt=PROMPT, raw_dir=tmp_path / "raw",
            provisioner=mock_provisioner(write_mock_harness(tmp_path)),
        )  # fmt: skip
        docker = _SpyCli()
        start = datetime.now(UTC)
        doc = execute_run(request, docker=docker)
        d = tmp_path / "raw" / "swebench" / plan.task.instance_id / doc.run_id

        assert doc.state is ExecutionState.TIMED_OUT  # the timeout outcome is preserved
        assert doc.timeout_action == "killed" and doc.cleanup.status == "ok"
        assert (datetime.now(UTC) - start).total_seconds() < 120  # not the 600 s sleep
        assert docker.survivors == []  # the agent process was really stopped, not just abandoned
        assert (d / "model.patch").exists() and (d / "git-status.txt").exists()  # evidence kept
        assert json.loads((d / "run.json").read_text())["timeout_action"] == "killed"
        # the unrelated container was never touched
        assert sh("docker", "inspect", "-f", "{{.State.Running}}", bystander) == "true"
        assert not [c for c in managed("container") if c]  # nothing of ours remains
    finally:
        sh("docker", "rm", "-f", bystander, check=False)


def test_no_timeout_plan_runs_without_any_wrapper(tmp_path, synthetic_task):
    doc, _ = run_mock(tmp_path, synthetic_task, timeout=None)
    assert doc.state is ExecutionState.COMPLETED and doc.plan.timeout_seconds is None


def test_restricted_provider_run_through_the_pinned_proxy(tmp_path, synthetic_task):
    from agent_harness_lab.execution.network import SQUID_IMAGE

    if not have_image(SQUID_IMAGE):
        pytest.skip("pinned Squid image not present locally")
    doc, d = run_mock(
        tmp_path,
        synthetic_task,
        network=NetworkMode.RESTRICTED_PROVIDER,
        allowed_hosts=["llm.example.test"],
    )
    assert doc.state is ExecutionState.COMPLETED, doc.error
    assert (d / "proxy.log").exists()  # collected before the proxy was removed
    assert doc.network.squid_policy_sha256 and doc.network.allowed_hosts == ["llm.example.test"]


# ---- restricted-provider network behaviour, fully offline ----
def test_restricted_network_allows_only_the_allowlisted_host(tmp_path):
    from agent_harness_lab.execution.models import ContainerSpec
    from agent_harness_lab.execution.network import SQUID_IMAGE, proxy_env

    for needed in (SQUID_IMAGE, "curlimages/curl:latest", "nginx:latest"):
        if not have_image(needed):
            pytest.skip(f"{needed} not present locally")

    docker = DockerCli()
    tag = uuid.uuid4().hex[:8]
    labels = {"ahl.managed": "true", "ahl.run-id": f"nettest-{tag}"}
    allowed = "allowed.example.test"
    res = plan_network(
        NetworkPlan(mode=NetworkMode.RESTRICTED_PROVIDER, allowed_hosts=[allowed]),
        names=(f"ahl-net-nettest-{tag}", f"ahl-proxy-nettest-{tag}"),
    )
    origin_net, origin, client = (
        f"ahl-test-origin-{tag}",
        f"ahl-test-nginx-{tag}",
        f"ahl-test-client-{tag}",
    )
    try:
        start_network(docker, res, labels=labels, staging_dir=tmp_path)
        # A local "internet": nginx reachable (by the allowed name) only from the proxy's side.
        sh("docker", "network", "create", "--label", "ahl.managed=true", origin_net)
        sh("docker", "run", "-d", "--name", origin, "--network", origin_net,
           "--network-alias", allowed, "--label", "ahl.managed=true", "nginx:latest")  # fmt: skip
        sh("docker", "network", "connect", origin_net, res.record.proxy_name)
        docker.create_container(
            ContainerSpec(
                name=client, image="curlimages/curl:latest", network=res.record.network_name,
                env=proxy_env(), pass_env=[], mounts=[], labels=labels,
            )
        )  # fmt: skip
        docker.start(client)

        def curl(*args):
            return docker.exec(
                client,
                ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", "20", *args],
            )

        proxy = ["-x", "http://squid:3128"]
        ok = curl(*proxy, f"http://{allowed}/")
        assert (ok.returncode, ok.stdout) == (0, "200")  # allowlisted host: through the proxy
        denied = curl(*proxy, "http://blocked.example.test/")
        assert denied.stdout == "403"  # any other host: denied by the proxy policy
        direct_ip = curl("--noproxy", "*", "--max-time", "5", "http://1.1.1.1/")
        assert direct_ip.returncode != 0  # --internal network: no direct route to the internet
        direct_origin = curl("--noproxy", "*", "--max-time", "5", f"http://{allowed}/")
        assert (
            direct_origin.returncode != 0
        )  # even the allowed host is unreachable except via proxy
    finally:
        sh("docker", "rm", "-f", client, origin, res.record.proxy_name, check=False)
        sh("docker", "network", "rm", origin_net, check=False)
        sh("docker", "network", "rm", res.record.network_name, check=False)
    assert not [c for c in managed("container") if tag in c]  # nothing from this test remains
