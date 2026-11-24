"""Tool classification, exactly as in the frozen v1 extractor.

Known imperfections (documented, deliberately NOT fixed in ahl-metrics-v1):
search/list tools such as OpenCode ``grep``/``glob`` and any unrecognised tool
(e.g. ``execute``) fall into "other"; ``patch``/``apply_patch`` count as edits.
"""

from __future__ import annotations

from typing import Literal

ToolClass = Literal["read", "bash", "edit", "write", "other"]


def classify_tool(name: str | None) -> ToolClass:
    if name == "read":
        return "read"
    if name in {"bash", "shell"}:
        return "bash"
    if name in {"edit", "patch", "apply_patch"}:
        return "edit"
    if name == "write":
        return "write"
    return "other"
