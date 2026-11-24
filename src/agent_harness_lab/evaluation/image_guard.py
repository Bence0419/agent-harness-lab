"""Protect the shared task-image tag around an official evaluation (approach C).

The official evaluator, given ``--task-repo``, force-rebuilds the image named in ``task.yaml``:
it removes the existing tag and builds a new image under it (swebench 5.0.2
``run_evaluation._build_before_eval`` -> ``build_instance_images(force_rebuild=True)``). There
is no supported existing-image mode or alternative namespace for ``swebench eval``, and
omitting ``--task-repo`` would change the TEST SOURCE (the published dataset). So the evaluator
runs unchanged and AHL guards the tag around it:

1. ``snapshot``: point an AHL-controlled reference at the current image id. Because the
   evaluator removes images by tag name, this reference keeps the original image alive.
2. the evaluator runs;
3. ``restore``: record the evaluator-produced image id, re-point the original tag at its
   original id, verify, and only then drop the snapshot reference. The snapshot is KEPT if
   restoration cannot be verified;
4. (caller) verify the evaluator image's baseline by its ID, then ``release_evaluator_image``:
   remove the evaluator-built image by default, or keep it under an AHL reference on request.

Restoration never raises: failures are returned as evidence so they can be reported prominently.
No image that existed before the evaluation is ever deleted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from agent_harness_lab.execution.docker import DockerClient
from agent_harness_lab.execution.models import DockerError

SNAPSHOT_NAMESPACE = "ahl-eval-snapshot"
EVALUATOR_IMAGE_NAMESPACE = "ahl-eval-image"
Restoration = Literal["not-needed", "restored", "failed", "skipped"]
Retention = Literal[
    "not-applicable",  # no separate evaluator image exists
    "removed",  # default: the evaluator image created by this evaluation was removed
    "kept",  # --keep-evaluator-image: kept under an AHL reference
    "retained-for-recovery",  # restoration failed: nothing is cleaned up
    "skipped-preexisting",  # its id existed before the evaluation: never touched
    "skipped-unsafe",  # could not establish what existed before: not removed
    "remove-failed",  # removal was refused/failed: left in place
]


def _repo_name(reference: str) -> str:
    """Last path component of an image reference without tag/digest, lowercased and safe."""
    name = reference.rsplit("/", 1)[-1].split("@", 1)[0].split(":", 1)[0].lower()
    return re.sub(r"[^a-z0-9._-]", "-", name) or "image"


def _tag_part(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "-", text)[:100].lstrip(".-") or "x"


@dataclass
class ImageOutcome:
    image_id_before: str | None = None
    evaluator_image_id: str | None = None
    final_image_id: str | None = None
    restoration: Restoration = "skipped"
    snapshot_ref: str | None = None
    snapshot_removed: bool = False
    evaluator_image_ref: str | None = None  # only when retention == "kept"
    retention: Retention = "not-applicable"
    rebuilt: bool = False
    warnings: list[str] = field(default_factory=list)


class ImageGuard:
    def __init__(self, docker: DockerClient, reference: str, label: str) -> None:
        self._docker = docker
        self._reference = reference
        self._label = _tag_part(label)
        self._before: str | None = None
        self._snapshot: str | None = None
        self._existing_ids: set[str] | None = None  # every image that existed before

    def snapshot(self) -> str | None:
        """Pin the current image id. Raises DockerError if the tag cannot be protected, in
        which case the evaluator must NOT be started."""
        self._before = self._docker.image_inspect(self._reference).image_id
        if self._before is None:
            raise DockerError(f"cannot determine the image id of {self._reference}")
        try:
            self._existing_ids = self._docker.list_image_ids()
        except DockerError:
            self._existing_ids = None  # unknown: the evaluator image will then never be removed
        self._snapshot = f"{SNAPSHOT_NAMESPACE}/{_repo_name(self._reference)}:{self._label}"
        self._docker.tag_image(self._before, self._snapshot)
        return self._before

    def restore(self) -> ImageOutcome:
        out = ImageOutcome(image_id_before=self._before, snapshot_ref=self._snapshot)
        if self._before is None:  # snapshot never taken: nothing was changed by us
            return out
        try:
            out.evaluator_image_id = self._docker.image_inspect(self._reference).image_id
        except DockerError:
            out.evaluator_image_id = None  # the tag is gone (evaluator removed it, build failed)
        out.rebuilt = out.evaluator_image_id != self._before

        if not out.rebuilt:
            out.final_image_id, out.restoration = out.evaluator_image_id, "not-needed"
        else:
            try:
                self._docker.tag_image(self._before, self._reference)
                out.final_image_id = self._docker.image_inspect(self._reference).image_id
            except DockerError as exc:
                out.final_image_id = None
                out.warnings.append(f"restoring {self._reference} failed: {exc}")
            out.restoration = "restored" if out.final_image_id == self._before else "failed"

        if out.restoration == "failed":
            out.warnings.insert(
                0,
                f"TASK IMAGE TAG NOT RESTORED: {self._reference} should point to "
                f"{self._before} but points to {out.final_image_id}. The original image is "
                f"preserved as {self._snapshot}; restore it with: docker tag {self._before} "
                f"{self._reference}",
            )
            return out  # keep the snapshot reference: it is the only safe handle
        try:  # the original tag again references the image, so this cannot delete it
            if self._snapshot:
                self._docker.untag_image(self._snapshot)
                out.snapshot_removed = True
        except DockerError as exc:
            out.warnings.append(f"could not remove temporary reference {self._snapshot}: {exc}")
        return out

    def release_evaluator_image(self, out: ImageOutcome, *, keep: bool) -> None:
        """After provenance checks: dispose of the evaluator-built image (default) or keep it.

        Only an image that THIS evaluation created is ever removed: its id must differ from the
        original and must not have existed before the evaluation, and removal is a plain
        ``docker rmi <id>`` that Docker refuses while any tag still references it. No pruning.
        If restoration failed, nothing is cleaned up (recoverability first).
        """
        if out.restoration == "failed":
            out.retention = "retained-for-recovery"
            return
        image_id = out.evaluator_image_id
        if not out.rebuilt or image_id is None:
            out.retention = "not-applicable"
            return
        if keep:
            ref = f"{EVALUATOR_IMAGE_NAMESPACE}/{_repo_name(self._reference)}:" + _tag_part(
                image_id.removeprefix("sha256:")[:12]
            )
            try:
                self._docker.tag_image(image_id, ref)
                out.evaluator_image_ref, out.retention = ref, "kept"
            except DockerError as exc:
                out.retention = "remove-failed"
                out.warnings.append(f"could not keep the evaluator image: {exc}")
            return
        if self._existing_ids is None:
            out.retention = "skipped-unsafe"
            out.warnings.append("evaluator image not removed: pre-existing images are unknown")
            return
        if image_id in self._existing_ids:
            out.retention = "skipped-preexisting"
            return
        try:
            self._docker.remove_image(image_id)
            out.retention = "removed"
        except DockerError as exc:
            out.retention = "remove-failed"
            out.warnings.append(f"evaluator image {image_id[:19]} could not be removed: {exc}")
