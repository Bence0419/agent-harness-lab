"""In-memory DockerClient for executor tests. It records every call and simulates just enough
of a task container (git commands, harness output) to exercise the execution layer."""

from __future__ import annotations

from pathlib import Path

from agent_harness_lab.execution.models import (
    ContainerSpec,
    DockerError,
    ExecOutput,
    ImageMissing,
    ImageRecord,
    Provision,
)
from agent_harness_lab.execution.network import SQUID_IMAGE

PATCH_TEXT = (
    "diff --git a/mock_target.txt b/mock_target.txt\n"
    "--- a/mock_target.txt\n+++ b/mock_target.txt\n@@ -1 +1,2 @@\n"
    " original\n+edited by mock harness\n"
)


class FakeDocker:
    def __init__(
        self,
        *,
        base_commit: str = "a" * 40,
        images: set[str] | None = None,
        harness_rc: int = 0,
        harness_timed_out: bool = False,
        patch_text: str = PATCH_TEXT,
        porcelain: str = "",
        head: str | None = None,
        proxy_log: str = "proxy log line\n",
        fail_on: dict[str, Exception] | None = None,
    ) -> None:
        self.base_commit = base_commit
        self.images = images if images is not None else {"ahl-test-synthetic:1", SQUID_IMAGE}
        self.image_ids: dict[str, str] = {}  # reference -> image id (default: one shared id)
        self.deleted_ids: list[str] = []  # image ids that lost their LAST reference
        self.known_ids: set[str] = set()  # unreferenced (dangling) images that exist
        self.baselines: dict[str, dict] = {}  # image id -> /testbed state (default: healthy)
        self.harness_rc = harness_rc
        self.harness_timed_out = harness_timed_out
        self.patch_text = patch_text
        self.porcelain = porcelain
        self.head = head or base_commit
        self.proxy_log = proxy_log
        self.fail_on = fail_on or {}
        self.calls: list[tuple[str, tuple]] = []
        self.networks: set[str] = set()
        self.containers: dict[str, ContainerSpec] = {}
        self.proxies: set[str] = set()
        self.exec_log: list[dict] = []
        self.copies: list[tuple[str, str, str]] = []
        self.proxy_mounts: list[Provision] = []
        self.on_kill = None  # optional hook called when a container is killed

    # -- helpers -------------------------------------------------------------------
    def _record(self, op: str, *args) -> None:
        self.calls.append((op, args))
        if op in self.fail_on:
            raise self.fail_on[op]

    def ops(self) -> list[str]:
        return [op for op, _ in self.calls]

    # -- DockerClient -------------------------------------------------------------------
    DEFAULT_ID = "sha256:" + "1" * 64

    def _id_of(self, ref: str) -> str | None:
        if ref in self.image_ids:
            return self.image_ids[ref]
        if ref.startswith("sha256:"):
            return ref
        return self.DEFAULT_ID if ref in self.images else None

    def image_inspect(self, reference: str) -> ImageRecord:
        self._record("image_inspect", reference)
        image_id = self._id_of(reference)
        if image_id is None or (
            reference not in self.images and not reference.startswith("sha256:")
        ):
            raise ImageMissing(f"Docker image not found locally: {reference}")
        return ImageRecord(reference=reference, image_id=image_id, repo_digests=[])

    def _referenced(self, image_id: str) -> bool:
        return any(self._id_of(r) == image_id for r in self.images)

    def tag_image(self, source: str, target: str) -> None:
        self._record("tag_image", source, target)
        image_id = self._id_of(source)
        if image_id is None:
            raise DockerError(f"no such image: {source}")
        self.image_ids[target] = image_id
        self.images.add(target)

    def untag_image(self, reference: str) -> None:
        self._record("untag_image", reference)
        if reference not in self.images:
            raise DockerError(f"no such image: {reference}")
        image_id = self._id_of(reference)
        self.images.discard(reference)
        self.image_ids.pop(reference, None)
        if image_id and not self._referenced(image_id):
            self.deleted_ids.append(image_id)  # `docker rmi` of the last reference deletes it
            self.known_ids.discard(image_id)

    def external_rebuild(self, reference: str, new_id: str) -> None:
        """What the official evaluator does: remove the tag, then build a new image under it."""
        old = self._id_of(reference)
        self.images.discard(reference)
        self.image_ids.pop(reference, None)
        if old and not self._referenced(old):
            self.deleted_ids.append(old)
            self.known_ids.discard(old)
        self.images.add(reference)
        self.image_ids[reference] = new_id
        self.known_ids.add(new_id)

    def list_image_ids(self) -> set[str]:
        self._record("list_image_ids")
        return {i for r in self.images if (i := self._id_of(r))} | set(self.known_ids)

    def remove_image(self, image_id: str) -> None:
        self._record("remove_image", image_id)
        if self._referenced(image_id):
            raise DockerError(f"image {image_id[:19]} is referenced by a tag")
        if image_id not in self.known_ids:
            raise DockerError(f"no such image: {image_id[:19]}")
        self.known_ids.discard(image_id)
        self.deleted_ids.append(image_id)

    def _baseline_exec(self, image_id: str, argv: list[str]) -> ExecOutput:
        """The git/test commands AHL runs to verify an evaluator image's /testbed."""
        st = {
            "exists": True,
            "git": True,
            "head": self.base_commit,
            "porcelain": "",
            "git_error": False,
        }
        st.update(self.baselines.get(image_id, {}))
        if argv[0] == "test":
            return ExecOutput(returncode=0 if st["exists"] else 1)
        if st["git_error"] or not st["git"]:
            return ExecOutput(returncode=128, stderr="fatal: not a git repository")
        if "--is-inside-work-tree" in argv:
            return ExecOutput(returncode=0, stdout="true\n")
        if "HEAD" in argv:
            return ExecOutput(returncode=0, stdout=st["head"] + "\n")
        return ExecOutput(returncode=0, stdout=st["porcelain"])  # status --porcelain

    def list_image_refs(self) -> set[str]:
        self._record("list_image_refs")
        return set(self.images)

    def network_create(self, name, *, internal, labels):
        self._record("network_create", name, internal, dict(labels))
        self.networks.add(name)

    def network_connect(self, network, container):
        self._record("network_connect", network, container)

    def network_remove(self, name):
        self._record("network_remove", name)
        self.networks.discard(name)

    def run_proxy(self, *, name, network, alias, image, mounts, labels):
        self._record("run_proxy", name, network, alias, image)
        self.proxies.add(name)
        self.proxy_mounts = list(mounts)

    def create_container(self, spec: ContainerSpec) -> None:
        self._record("create_container", spec.name)
        self.containers[spec.name] = spec

    def start(self, name):
        self._record("start", name)

    def kill(self, name):
        self._record("kill", name)
        if self.on_kill:
            self.on_kill()

    def inspect_container(self, name):
        self._record("inspect_container", name)
        return {"Id": "container-id-" + name}

    def copy_in(self, source: Path, container, destination):
        self._record("copy_in", str(source), container, destination)
        self.copies.append((str(source), container, destination))

    def exec(
        self, container, argv, *, workdir=None, timeout=None, stdout_path=None, stderr_path=None
    ):
        self._record("exec", tuple(argv))
        self.exec_log.append({"argv": list(argv), "workdir": workdir, "timeout": timeout})
        if argv[0] == "test" or "-C" in argv:  # baseline verification of an evaluator image
            image = self.containers[container].image if container in self.containers else ""
            return self._baseline_exec(self._id_of(image) or image, argv)
        if argv[0] == "git" or argv[:2] == ["git", "-c"]:
            return self._git(argv, stdout_path)
        if argv[0] == "mkdir":
            return ExecOutput(returncode=0)
        # anything else is the harness invocation
        if stdout_path:
            stdout_path.write_text('{"type":"mock_start"}\n{"type":"mock_end"}\n')
        if stderr_path:
            stderr_path.write_text("")
        return ExecOutput(returncode=self.harness_rc, timed_out=self.harness_timed_out)

    def _git(self, argv, stdout_path) -> ExecOutput:
        sub = [a for a in argv[1:] if not a.startswith("core.fileMode") and a != "-c"]
        if sub[:1] == ["rev-parse"]:
            return ExecOutput(returncode=0, stdout=self.head + "\n")
        if sub[:2] == ["status", "--porcelain"]:
            return ExecOutput(returncode=0, stdout=self.porcelain)
        if sub[:2] == ["status", "--short"]:
            if stdout_path:
                stdout_path.write_text(" M mock_target.txt\n" if self.patch_text else "")
            return ExecOutput(returncode=0)
        if sub[:1] == ["diff"]:
            if stdout_path:
                stdout_path.write_text(self.patch_text)
            return ExecOutput(returncode=0)
        return ExecOutput(returncode=0)  # config, reset, clean, add

    def logs(self, name):
        self._record("logs", name)
        return self.proxy_log

    def remove(self, *names):
        self._record("remove", names)
        for n in names:
            self.containers.pop(n, None)
            self.proxies.discard(n)


def docker_error(msg: str) -> DockerError:
    return DockerError(msg)
