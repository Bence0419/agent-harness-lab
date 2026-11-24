"""Provisioning parity: the mounts/copies the adapters declare must equal what the frozen
runners mounted or copied for the in-container executable paths to work. Nothing is executed."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agent_harness_lab.harnesses import HarnessError, get_harness
from agent_harness_lab.paths import find_project_root

ROOT = find_project_root()
pytestmark = pytest.mark.skipif(ROOT is None, reason="needs a source checkout with scripts/")

NODE = "/home/u/.nvm/versions/node/v24.19.0/bin/node"
PROJECT = "/proj"


class FakeHost:
    def __init__(self, links: dict[str, str] | None = None, node: str | None = NODE):
        self.node = node
        self.links = links or {
            f"{PROJECT}/node_modules/.bin/pi": (
                f"{PROJECT}/node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js"
            ),
            f"{PROJECT}/node_modules/.bin/opencode": (
                f"{PROJECT}/node_modules/@opencode/cli/bin/opencode.exe"
            ),
        }

    def which(self, name):
        return self.node if name == "node" else None

    def realpath(self, path):
        return self.links.get(path, path)

    missing: frozenset = frozenset()

    def is_file(self, path):
        return path not in self.missing

    def is_dir(self, path):
        return path not in self.missing


def frozen_agent_mounts(script: str) -> dict[str, tuple[str, bool]]:
    """{host variable: (container destination, read_only)} for `-v "$VAR:/dest[:ro]"` lines."""
    out = {}
    for var, dest, ro in re.findall(r'-v "\$(\w+):([^":]+)(:ro)?"', script):
        if var != "SQUID_CONFIG":  # the proxy's mount, not the agent's
            out[var] = (dest, bool(ro))
    return out


def script(name: str) -> str:
    return (ROOT / "scripts" / name).read_text()


STAGED_PI = {"/root/.pi/agent/models.json": Path("/run/generated/00-models.json")}
STAGED_OC = {"/etc/opencode/opencode.json": Path("/run/generated/00-opencode.json")}


@pytest.mark.parametrize(
    "name", ["run-swebench-task-astra.sh", "run-swebench-task.sh", "run-swebench-task-bme-qwen.sh"]
)
def test_pi_bind_mounts_match_frozen_runners(name):
    frozen = frozen_agent_mounts(script(name))
    prov = get_harness("pi").provisioning(f"{PROJECT}/node_modules/.bin/pi", FakeHost(), {})
    mine = {(p.destination, p.read_only) for p in prov if p.kind == "bind-mount"}
    expected = {frozen["HOST_NODE"], frozen["PI_MODULES"]}
    assert mine == expected == {("/opt/node24", True), ("/opt/pi-node_modules", True)}


def test_pi_sources_follow_the_frozen_variables():
    prov = get_harness("pi").provisioning(f"{PROJECT}/node_modules/.bin/pi", FakeHost(), {})
    src = {p.destination: p.source for p in prov}
    assert src["/opt/node24"] == "/home/u/.nvm/versions/node/v24.19.0"  # HOST_NODE: the Node root
    assert src["/opt/pi-node_modules"] == f"{PROJECT}/node_modules"  # PI_MODULES


def test_pi_config_is_a_writable_copy_at_the_frozen_cp_destination():
    text = script("run-swebench-task-bme-qwen.sh")
    frozen = frozen_agent_mounts(text)
    assert frozen["PI_BME_CONFIG"] == ("/tmp/pi-models.json", True)  # mounted read-only at /tmp...
    (cp_dest,) = re.findall(r"cp /tmp/pi-models\.json (\S+)", text)  # ...then copied into place
    prov = get_harness("pi").provisioning(f"{PROJECT}/node_modules/.bin/pi", FakeHost(), STAGED_PI)
    (copy,) = [p for p in prov if p.kind == "copy-in"]
    assert (copy.destination, copy.read_only) == (cp_dest, False)
    assert copy.source == "/run/generated/00-models.json"


@pytest.mark.parametrize(
    "name", ["run-swebench-task-opencode-astra.sh", "run-swebench-task-opencode.sh",
             "run-swebench-task-opencode-bme-qwen.sh"],
)  # fmt: skip
def test_opencode_mounts_match_frozen_runners(name):
    frozen = frozen_agent_mounts(script(name))
    prov = get_harness("opencode").provisioning(
        f"{PROJECT}/node_modules/.bin/opencode", FakeHost(), STAGED_OC
    )
    assert all(p.kind == "bind-mount" and p.read_only for p in prov)
    mine = {(p.destination, p.read_only) for p in prov}
    assert mine == {frozen["OPENCODE_BIN"], frozen["OPENCODE_POLICY"]}
    src = {p.destination: p.source for p in prov}
    # `readlink -f node_modules/.bin/opencode`: the symlink-resolved binary
    assert (
        src["/usr/local/bin/opencode"] == f"{PROJECT}/node_modules/@opencode/cli/bin/opencode.exe"
    )
    assert src["/etc/opencode/opencode.json"] == "/run/generated/00-opencode.json"


def test_provisioning_destinations_are_the_invocation_paths():
    from agent_harness_lab.harnesses.opencode import CONTAINER_EXECUTABLE as OC_EXE
    from agent_harness_lab.harnesses.pi import CONTAINER_EXECUTABLE as PI_EXE

    assert PI_EXE.startswith("/opt/pi-node_modules/")
    assert OC_EXE == "/usr/local/bin/opencode"


def test_the_task_directory_is_never_part_of_any_provisioning():
    for name, exe in (("pi", "pi"), ("opencode", "opencode")):
        prov = get_harness(name).provisioning(
            f"{PROJECT}/node_modules/.bin/{exe}", FakeHost(), STAGED_PI | STAGED_OC
        )
        for p in prov:
            assert "swe-bench" not in p.source and "tasks/" not in p.source


def test_pi_without_host_node_fails_clearly():
    with pytest.raises(HarnessError, match="needs Node.js"):
        get_harness("pi").provisioning(f"{PROJECT}/node_modules/.bin/pi", FakeHost(node=None), {})


def test_pi_outside_node_modules_fails_clearly():
    host = FakeHost(links={"/usr/bin/pi": "/usr/bin/pi"})
    with pytest.raises(HarnessError, match="cannot determine the node_modules"):
        get_harness("pi").provisioning("/usr/bin/pi", host, {})
