"""Harness executable/version discovery shared by ``ahl doctor`` and ``ahl run``.

Order: 1. PATH, 2. ``<project-root>/node_modules/.bin/<executable>``. A global install
therefore takes precedence over the project-local one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal, Protocol

Source = Literal["PATH", "project-local"]


class DiscoveryProbe(Protocol):
    def which(self, name: str) -> str | None: ...
    def is_executable(self, path: str) -> bool: ...
    def read_text(self, path: str) -> str | None: ...
    def project_root(self) -> str | None: ...


@dataclass(frozen=True)
class Discovered:
    path: str
    source: Source


def parse_version(text: str) -> str | None:
    match = re.search(r"\d+(?:\.\d+)+", text)
    return match.group(0) if match else None


def discover_executable(probe: DiscoveryProbe, executable: str) -> Discovered | None:
    on_path = probe.which(executable)
    if on_path is not None:
        return Discovered(on_path, "PATH")
    root = probe.project_root()
    if root is not None:
        local = f"{root}/node_modules/.bin/{executable}"
        if probe.is_executable(local):
            return Discovered(local, "project-local")
    return None


def pinned_version(probe: DiscoveryProbe, package: str) -> str | None:
    """Version pinned for ``package`` in the project's package.json, if available."""
    root = probe.project_root()
    if root is None:
        return None
    text = probe.read_text(f"{root}/package.json")
    if text is None:
        return None
    try:
        spec = json.loads(text).get("dependencies", {}).get(package)
    except (ValueError, AttributeError):
        return None
    return parse_version(spec) if isinstance(spec, str) else None
