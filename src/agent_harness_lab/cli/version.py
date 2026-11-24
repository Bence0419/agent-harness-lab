import typer
from rich.console import Console

from agent_harness_lab import __version__


def version_command() -> None:
    """Print the Agent Harness Lab version."""
    Console(highlight=False).print(f"Agent Harness Lab {__version__}")
    raise typer.Exit(0)
