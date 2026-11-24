"""Streaming JSONL reader that tolerates damaged lines and never modifies the file."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MAX_KEPT_WARNINGS = 20


@dataclass(frozen=True)
class TraceWarning:
    kind: str  # "malformed_line" | "truncated_final_line" | "non_object_line" | "missing_file"
    line: int | None
    message: str


@dataclass
class TraceReport:
    """Filled in progressively while the trace is consumed."""

    path: Path
    exists: bool = True
    lines: int = 0  # non-blank lines seen
    objects: int = 0  # lines that parsed to a JSON object
    malformed_lines: int = 0
    truncated_final_line: bool = False
    warning_count: int = 0
    warnings: list[TraceWarning] = field(default_factory=list)  # first MAX_KEPT_WARNINGS

    @property
    def usable(self) -> bool:
        """True if at least one JSON object was read (an empty trace is not usable)."""
        return self.exists and self.objects > 0

    def add(self, warning: TraceWarning) -> None:
        self.warning_count += 1
        if len(self.warnings) < MAX_KEPT_WARNINGS:
            self.warnings.append(warning)

    def summary(self) -> str | None:
        """One-line description of problems, or None if the trace was clean."""
        parts: list[str] = []
        if not self.exists:
            parts.append("trace.jsonl missing")
        elif self.objects == 0:
            parts.append("empty trace" if self.lines == 0 else "no valid JSON events")
        if self.malformed_lines:
            first = next((w.line for w in self.warnings if w.kind == "malformed_line"), None)
            where = f", first at line {first}" if first else ""
            parts.append(f"{self.malformed_lines} malformed line(s) skipped{where}")
        if self.truncated_final_line:
            parts.append("truncated final line ignored")
        other = self.warning_count - self.malformed_lines - int(self.truncated_final_line)
        if other > 0 and self.exists:
            parts.append(f"{other} non-object line(s) skipped")
        return "; ".join(parts) if parts else None


def iter_json_objects(path: Path, report: TraceReport) -> Iterator[dict[str, Any]]:
    """Yield JSON objects line by line; record every problem in ``report``."""
    if not path.is_file():
        report.exists = False
        report.add(TraceWarning("missing_file", None, f"{path.name} not found"))
        return
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for number, raw in enumerate(f, start=1):
            if not raw.strip():
                continue
            report.lines += 1
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                if not raw.endswith("\n"):
                    report.truncated_final_line = True
                    report.add(TraceWarning("truncated_final_line", number, str(exc)))
                else:
                    report.malformed_lines += 1
                    report.add(TraceWarning("malformed_line", number, str(exc)))
                continue
            if not isinstance(obj, dict):
                report.add(TraceWarning("non_object_line", number, "JSON value is not an object"))
                continue
            report.objects += 1
            yield obj
