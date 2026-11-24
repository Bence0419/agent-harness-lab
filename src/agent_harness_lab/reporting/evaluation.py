from __future__ import annotations

from rich.console import Console
from rich.markup import escape

from agent_harness_lab.evaluation.models import Classification, EvaluationDocument

_STYLE = {
    Classification.BENCHMARK_PASS: "[bold green]PASS[/] (resolved)",
    Classification.BENCHMARK_FAIL: "[bold red]FAIL[/] (not resolved)",
    Classification.EVALUATION_FAILURE: (
        "[bold yellow]NO VERDICT[/] (evaluation infrastructure failure)"
    ),
    Classification.AMBIGUOUS: "[bold yellow]NO VERDICT[/] (ambiguous evaluator output)",
}


def render_evaluation(doc: EvaluationDocument, console: Console) -> None:
    def row(label: str, value: str) -> None:
        console.print(f"[bold]{label + ':':<27}[/]{escape(value)}")

    row("Run", doc.run_id)
    row("Task", doc.task["instance_id"])
    console.print(f"[bold]{'Result:':<27}[/]{_STYLE[doc.classification]}")
    row("State", doc.state.value)
    row("Evaluator", f"{doc.evaluator} (swebench {doc.evaluator_version})")
    row("Evaluation duration", f"{doc.duration_seconds} s (separate from agent duration)")
    row("Patch sha256", doc.patch_sha256[:16] + "…")
    img = doc.image

    def short(i: str | None) -> str:
        return (i or "unknown").removeprefix("sha256:")[:12]

    row(
        "Task image ids",
        f"agent {short(img.agent_image_id)} | tag before {short(img.image_id_before)} | "
        f"evaluator {short(img.evaluator_image_id)} | tag after {short(img.final_image_id)}",
    )
    row(
        "Image rebuilt",
        f"{'YES' if img.rebuilt else 'no'}; tag restoration: {img.restoration}; "
        f"evaluator image: {img.retention}",
    )
    b = doc.evaluator_image_baseline
    if b is not None:
        detail = f"{b.verification_status}"
        if b.actual_base_commit:
            detail += (
                f" (/testbed at {b.actual_base_commit[:12]}, expected {b.expected_base_commit[:12]}"
            )
            detail += f", worktree {'clean' if b.worktree_clean else 'DIRTY'})"
        elif b.error:
            detail += f": {b.error}"
        row("Evaluator image baseline", detail)
    if doc.error:
        row("Error", doc.error)
    for w in doc.warnings:
        console.print(f"[yellow]warning:[/] {escape(w)}")
    row("Artifacts", ", ".join(sorted(doc.artifacts)) or "none")
