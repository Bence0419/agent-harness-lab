import typer

from agent_harness_lab.cli.benchmark import benchmark_command, experiments_app
from agent_harness_lab.cli.compare import compare_command
from agent_harness_lab.cli.doctor import doctor_command
from agent_harness_lab.cli.evaluate import evaluate_command
from agent_harness_lab.cli.results import results_command
from agent_harness_lab.cli.run import run_command
from agent_harness_lab.cli.tasks import tasks_app
from agent_harness_lab.cli.version import version_command

app = typer.Typer(
    name="ahl",
    help="Agent Harness Lab: benchmark AI coding-agent harnesses reproducibly.",
    no_args_is_help=True,
    add_completion=False,
)

app.command("version")(version_command)
app.command("doctor")(doctor_command)
app.command("results")(results_command)
app.command("compare")(compare_command)
app.command("run")(run_command)
app.command("evaluate")(evaluate_command)
app.command("benchmark")(benchmark_command)
app.add_typer(experiments_app, name="experiments")
app.add_typer(tasks_app, name="tasks")

if __name__ == "__main__":
    app()
