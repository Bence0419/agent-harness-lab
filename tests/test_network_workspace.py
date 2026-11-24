import hashlib
import re

import pytest

from agent_harness_lab.execution.models import DockerError, StageError
from agent_harness_lab.execution.network import (
    PROXY_SETTLE_SECONDS,
    SQUID_IMAGE,
    plan_network,
    proxy_env,
    render_squid_policy,
    start_network,
)
from agent_harness_lab.execution.workspace import capture_patch, prepare_workspace
from agent_harness_lab.paths import find_project_root
from agent_harness_lab.runplan.models import NetworkMode, NetworkPlan
from tests.fake_docker import FakeDocker

ROOT = find_project_root()
R = NetworkPlan(mode=NetworkMode.RESTRICTED_PROVIDER, allowed_hosts=["llm.example.org"])
U = NetworkPlan(mode=NetworkMode.UNRESTRICTED, allowed_hosts=[])


# ---- Squid policy -----------------------------------------------------------------
def test_squid_policy_is_deterministic_and_host_order_independent():
    a = render_squid_policy(["b.example.org", "a.example.org", "a.example.org"])
    b = render_squid_policy(["a.example.org", "b.example.org"])
    assert a == b
    assert "acl provider_hosts dstdomain a.example.org b.example.org\n" in a


def test_squid_policy_allows_only_the_listed_hosts():
    policy = render_squid_policy(["llm.example.org"])
    allow = [ln for ln in policy.splitlines() if ln.startswith("http_access")]
    assert allow == [
        "http_access deny !Safe_ports",
        "http_access deny CONNECT !SSL_ports",
        "http_access allow provider_hosts",
        "http_access deny all",  # everything else is denied: no general internet access
    ]
    assert "cache deny all" in policy


def test_squid_policy_requires_a_host():
    with pytest.raises(ValueError, match="at least one allowed host"):
        render_squid_policy([])


@pytest.mark.skipif(ROOT is None, reason="needs the frozen configs")
@pytest.mark.parametrize(
    "conf", ["swebench-openai-only-squid-v1.conf", "swebench-bme-only-squid-v1.conf"]
)
def test_policy_matches_frozen_squid_configs_except_the_acl_name(conf):
    frozen = (ROOT / "configs" / conf).read_text()
    host = re.search(r"acl \w+ dstdomain (\S+)", frozen).group(1)
    normalized = re.sub(r"\b(openai_api|bme_llm)\b", "provider_hosts", frozen)
    assert render_squid_policy([host]) == normalized.rstrip("\n") + "\n"


@pytest.mark.skipif(ROOT is None, reason="needs the frozen runners")
def test_pinned_squid_image_and_proxy_env_match_frozen_runners():
    script = (ROOT / "scripts/run-swebench-task-opencode-astra.sh").read_text()
    assert f'SQUID_IMAGE="{SQUID_IMAGE}"' in script
    env = proxy_env()
    for key, value in env.items():
        assert f'-e {key}="{value}"' in script, key
    assert "--network-alias squid" in script and "docker network connect bridge" in script
    assert "sleep 2" in script and PROXY_SETTLE_SECONDS == 2.0


# ---- network plans ---------------------------------------------------------------------
def test_unrestricted_plan_has_no_proxy_network_or_env(tmp_path):
    res = plan_network(U, names=("n", "p"))
    assert res.record.mode == "unrestricted" and res.container_env == {} and res.policy is None
    d = FakeDocker()
    start_network(d, res, labels={}, staging_dir=tmp_path, sleep=lambda s: None)
    assert d.calls == []  # ordinary Docker networking: nothing is created
    assert res.network_name is None and res.proxy_name is None


def test_restricted_plan_records_policy_hash_and_hosts(tmp_path):
    res = plan_network(R, names=("net-x", "proxy-x"))
    assert res.record.allowed_hosts == ["llm.example.org"]
    assert res.record.squid_image == SQUID_IMAGE
    assert res.record.squid_policy_sha256 == hashlib.sha256(res.policy.encode()).hexdigest()
    assert res.container_env == proxy_env()


def test_restricted_start_reproduces_the_frozen_sequence(tmp_path):
    res = plan_network(R, names=("net-x", "proxy-x"))
    d, slept = FakeDocker(), []
    start_network(d, res, labels={"ahl.run-id": "r"}, staging_dir=tmp_path, sleep=slept.append)
    assert d.ops() == ["image_inspect", "network_create", "run_proxy", "network_connect"]
    _, (name, internal, labels) = d.calls[1]
    assert (name, internal) == ("net-x", True)  # --internal network
    _, (pname, network, alias, image) = d.calls[2]
    assert (pname, network, alias, image) == ("proxy-x", "net-x", "squid", SQUID_IMAGE)
    assert d.calls[3][1] == ("bridge", "proxy-x")  # only the proxy gets egress
    assert slept == [2.0]
    (m,) = d.proxy_mounts
    assert (m.destination, m.read_only) == ("/etc/squid/squid.conf", True)
    assert (tmp_path / "squid.conf").read_text() == res.policy
    assert m.sha256 == res.record.squid_policy_sha256
    assert (res.network_created, res.proxy_created) == (True, True)


def test_squid_image_missing_fails_the_network_stage_without_creating_anything(tmp_path):
    res = plan_network(R, names=("n", "p"))
    d = FakeDocker(images=set())
    with pytest.raises(StageError, match="not found locally") as e:
        start_network(d, res, labels={}, staging_dir=tmp_path, sleep=lambda s: None)
    assert e.value.stage == "network" and d.networks == set() and not res.network_created


def test_partial_network_failure_reports_what_exists_for_cleanup(tmp_path):
    res = plan_network(R, names=("n", "p"))
    d = FakeDocker(fail_on={"run_proxy": DockerError("boom")})
    with pytest.raises(StageError):
        start_network(d, res, labels={}, staging_dir=tmp_path, sleep=lambda s: None)
    assert res.network_created is True and res.proxy_created is False


# ---- workspace ---------------------------------------------------------------------------
BASE = "a" * 40


def test_workspace_preparation_reproduces_the_frozen_sequence():
    d = FakeDocker(base_commit=BASE)
    prepare_workspace(d, "c", BASE, "/testbed")
    assert [e["argv"] for e in d.exec_log] == [
        ["git", "config", "core.fileMode", "false"],
        ["git", "rev-parse", "HEAD"],
        ["git", "reset", "--hard", BASE],
        ["git", "clean", "-fd"],
        ["git", "status", "--porcelain"],
    ]
    assert {e["workdir"] for e in d.exec_log} == {"/testbed"}


def test_workspace_rejects_unexpected_head_before_resetting():
    d = FakeDocker(base_commit=BASE, head="f" * 40)
    with pytest.raises(StageError, match="unexpected repository HEAD"):
        prepare_workspace(d, "c", BASE, "/testbed")
    assert ["git", "reset", "--hard", BASE] not in [e["argv"] for e in d.exec_log]


def test_workspace_rejects_a_dirty_worktree_after_cleaning():
    d = FakeDocker(base_commit=BASE, porcelain=" M leftover.txt\n")
    with pytest.raises(StageError, match="not clean before agent execution"):
        prepare_workspace(d, "c", BASE, "/testbed")


def test_patch_capture_commands_and_files(tmp_path):
    d = FakeDocker(base_commit=BASE, patch_text="PATCH\n")
    capture_patch(d, "c", BASE, "/testbed", tmp_path)
    argvs = [e["argv"] for e in d.exec_log]
    assert argvs == [
        ["git", "-c", "core.fileMode=false", "status", "--short"],
        ["git", "add", "-N", "--", "."],  # intent-to-add so new files appear in the diff
        ["git", "-c", "core.fileMode=false", "diff", "--binary", BASE],
    ]
    assert (tmp_path / "model.patch").read_text() == "PATCH\n"
    assert (tmp_path / "git-status.txt").exists()
