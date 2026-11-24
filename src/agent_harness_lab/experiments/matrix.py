"""Deterministic matrix expansion: tasks x harnesses x models x repetitions."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from agent_harness_lab.experiments.models import NormalizedExperiment


class MatrixCell(BaseModel):
    model_config = ConfigDict(frozen=True)

    cell_id: str
    task: str
    harness: str
    provider: str
    model: str
    reasoning: str | None
    repetition: int  # 1-based

    @property
    def model_ref(self) -> str:
        return f"{self.provider}/{self.model}"


def cell_id(
    task: str, harness: str, provider: str, model: str, reasoning: str | None, repetition: int
) -> str:
    """Stable identifier: a pure function of the cell's coordinates (never of YAML order)."""
    return (
        f"task={task};harness={harness};model={provider}/{model};"
        f"reasoning={reasoning or 'none'};repetition={repetition}"
    )


def expand_matrix(exp: NormalizedExperiment) -> list[MatrixCell]:
    """The normalized experiment is already sorted, so iteration order IS the canonical order:
    task, then harness, then model, then repetition."""
    cells = [
        MatrixCell(
            cell_id=cell_id(task, harness, m.provider, m.model, m.reasoning, rep),
            task=task,
            harness=harness,
            provider=m.provider,
            model=m.model,
            reasoning=m.reasoning,
            repetition=rep,
        )
        for task in exp.tasks
        for harness in exp.harnesses
        for m in exp.models
        for rep in range(1, exp.repetitions + 1)
    ]
    return cells
