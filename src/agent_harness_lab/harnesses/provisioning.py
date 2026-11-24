"""Host-side facts needed to provision a harness into the task container."""

from __future__ import annotations

import os
import shutil
from typing import Protocol


class HostEnvironment(Protocol):
    def which(self, name: str) -> str | None: ...
    def realpath(self, path: str) -> str: ...
    def is_file(self, path: str) -> bool: ...
    def is_dir(self, path: str) -> bool: ...


class RealHost:
    def which(self, name: str) -> str | None:
        return shutil.which(name)

    def realpath(self, path: str) -> str:
        return os.path.realpath(path)

    def is_file(self, path: str) -> bool:
        return os.path.isfile(path)

    def is_dir(self, path: str) -> bool:
        return os.path.isdir(path)
