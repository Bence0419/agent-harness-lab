import sys

import pytest

from agent_harness_lab.execution.docker import DockerCli, RunResult, SubprocessRunner
from agent_harness_lab.execution.models import (
    ContainerSpec,
    DockerError,
    ImageMissing,
    Provision,
)


class RecordingRunner:
    """Records argv and replies from a script. No process is ever started."""

    def __init__(self, *replies: RunResult):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def run(self, argv, **kw):
        self.calls.append({"argv": argv, **kw})
        return self.replies.pop(0) if self.replies else RunResult(0)

    @property
    def argv(self) -> list[str]:
        return self.calls[-1]["argv"]


def cli(*replies):
    r = RecordingRunner(*replies)
    return DockerCli(r), r


def mount(src, dst, ro=True):
    return Provision(kind="bind-mount", source=src, destination=dst, read_only=ro, purpose="test")


def test_every_command_is_an_argv_list_never_a_shell_string():
    d, r = cli()
    d.network_create("n", internal=True, labels={"a": "b"})
    d.network_connect("bridge", "p")
    d.network_remove("n")
    d.start("c")
    d.remove("c", "p")
    d.exec("c", ["git", "status"], workdir="/testbed")
    d.copy_in(__import__("pathlib").Path("/x"), "c", "/y")
    for call in r.calls:
        assert isinstance(call["argv"], list) and all(isinstance(a, str) for a in call["argv"])
        assert call["argv"][0] == "docker"
        assert "sh" not in call["argv"][:3] and "-c" not in call["argv"][:3]


def test_image_inspect_argv_and_parsing():
    out = '[{"Id": "sha256:abc", "RepoDigests": ["x@sha256:def"]}]'
    d, r = cli(RunResult(0, stdout=out))
    rec = d.image_inspect("img:latest")
    assert r.argv == ["docker", "image", "inspect", "img:latest"]
    assert (rec.reference, rec.image_id, rec.repo_digests) == (
        "img:latest",
        "sha256:abc",
        ["x@sha256:def"],
    )


def test_missing_image_fails_clearly_without_pulling():
    d, r = cli(RunResult(1, stderr="No such image"))
    with pytest.raises(ImageMissing, match="not found locally.*img:latest"):
        d.image_inspect("img:latest")
    assert len(r.calls) == 1  # no pull/build attempted


def test_network_create_internal_and_labels():
    d, r = cli()
    d.network_create("net1", internal=True, labels={"ahl.run-id": "r1", "ahl.managed": "true"})
    assert r.argv == [
        "docker", "network", "create", "--internal",
        "--label", "ahl.managed=true", "--label", "ahl.run-id=r1", "net1",
    ]  # fmt: skip
    d.network_create("net2", internal=False, labels={})
    assert "--internal" not in r.argv


def test_proxy_run_argv():
    d, r = cli()
    d.run_proxy(
        name="p", network="n", alias="squid", image="sq@sha256:1",
        mounts=[mount("/h/squid.conf", "/etc/squid/squid.conf")], labels={"k": "v"},
    )  # fmt: skip
    assert r.argv == [
        "docker", "run", "-d", "--name", "p", "--network", "n", "--network-alias", "squid",
        "--label", "k=v", "-v", "/h/squid.conf:/etc/squid/squid.conf:ro", "sq@sha256:1",
    ]  # fmt: skip


def test_container_create_argv_structured_mounts_and_secret_names_only():
    d, r = cli()
    d.create_container(
        ContainerSpec(
            name="agent", image="img", network="n",
            env={"PATH": "/opt/bin:/bin", "HTTP_PROXY": "http://squid:3128"},
            pass_env=["OPENAI_API_KEY"],
            mounts=[mount("/h/node", "/opt/node24"), mount("/h/w", "/w", ro=False)],
            labels={"ahl.run-id": "r"},
        )
    )  # fmt: skip
    assert r.argv == [
        "docker", "create", "--name", "agent", "--entrypoint", "sleep",
        "--label", "ahl.run-id=r", "--network", "n",
        "-e", "PATH=/opt/bin:/bin", "-e", "HTTP_PROXY=http://squid:3128",
        "-e", "OPENAI_API_KEY",  # name only: the value is never placed in argv
        "-v", "/h/node:/opt/node24:ro", "-v", "/h/w:/w",
        "img", "infinity",
    ]  # fmt: skip


def test_container_without_network_uses_default_bridge():
    d, r = cli()
    d.create_container(
        ContainerSpec(name="a", image="i", network=None, env={}, pass_env=[], mounts=[], labels={})
    )
    assert "--network" not in r.argv


def test_exec_argv_and_timeout_pass_through(tmp_path):
    d, r = cli(RunResult(0, stdout="out"))
    out = d.exec("c", ["git", "rev-parse", "HEAD"], workdir="/testbed")
    assert r.argv == ["docker", "exec", "-w", "/testbed", "c", "git", "rev-parse", "HEAD"]
    assert out.stdout == "out" and out.returncode == 0

    d.exec("c", ["agent"], timeout=12.5, stdout_path=tmp_path / "o", stderr_path=tmp_path / "e")
    call = r.calls[-1]
    assert call["timeout"] == 12.5 and call["stdout_path"] == tmp_path / "o"

    d.exec("c", ["agent"])
    assert r.calls[-1]["timeout"] is None  # no wrapper unless asked


def test_exec_reports_timeout():
    d, _ = cli(RunResult(-15, timed_out=True))
    out = d.exec("c", ["agent"], timeout=1)
    assert out.timed_out and out.returncode == -15


def test_copy_logs_inspect_remove_start_argv(tmp_path):
    d, r = cli(
        RunResult(0), RunResult(0, stdout="log text"), RunResult(0, stdout='[{"Id": "cid"}]')
    )
    d.copy_in(tmp_path / "f", "c", "/dst/f")
    assert r.argv == ["docker", "cp", str(tmp_path / "f"), "c:/dst/f"]
    assert d.logs("p") == "log text"
    assert r.argv == ["docker", "logs", "p"] and r.calls[-1]["merge_stderr"] is True
    assert d.inspect_container("c") == {"Id": "cid"}
    assert r.argv == ["docker", "container", "inspect", "c"]
    d.remove("a", "b")
    assert r.argv == ["docker", "rm", "-f", "a", "b"]
    d.start("a")
    assert r.argv == ["docker", "start", "a"]


def test_failures_raise_docker_error_with_last_stderr_line():
    d, _ = cli(RunResult(125, stderr="Error response\nthe real reason"))
    with pytest.raises(DockerError, match="exit 125.*the real reason"):
        d.start("c")


# ---- real subprocess runner (our own interpreter only; no Docker) ---------------------
def test_subprocess_runner_captures_output_and_redirects_to_files(tmp_path):
    out = tmp_path / "o.txt"
    r = SubprocessRunner().run(
        [sys.executable, "-c", "import sys; print('hi'); print('err', file=sys.stderr)"],
        stdout_path=out,
        stderr_path=tmp_path / "e.txt",
    )
    assert r.returncode == 0 and not r.timed_out
    assert out.read_text() == "hi\n" and (tmp_path / "e.txt").read_text() == "err\n"
    cap = SubprocessRunner().run([sys.executable, "-c", "print('x')"])
    assert cap.stdout == "x\n"


def test_subprocess_runner_applies_timeout_with_sigterm():
    r = SubprocessRunner().run([sys.executable, "-c", "import time; time.sleep(60)"], timeout=0.5)
    assert r.timed_out is True and r.returncode != 0


def test_subprocess_runner_without_timeout_waits_for_completion():
    r = SubprocessRunner().run([sys.executable, "-c", "import time; time.sleep(0.3)"], timeout=None)
    assert r.timed_out is False and r.returncode == 0


def test_kill_argv_targets_exactly_the_named_container():
    d, r = cli()
    d.kill("ahl-agent-run1")
    assert r.argv == ["docker", "kill", "ahl-agent-run1"]
