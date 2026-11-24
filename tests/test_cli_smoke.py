import pytest
from typer.testing import CliRunner

from agent_harness_lab import __version__
from agent_harness_lab.cli import doctor as doctor_cli
from agent_harness_lab.cli.app import app

runner = CliRunner()


@pytest.fixture
def use_probe(monkeypatch):
    def install(probe):
        monkeypatch.setattr(doctor_cli, "_make_probe", lambda: probe)

    return install


def test_help_lists_initial_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "version" in result.output
    assert "doctor" in result.output
    assert {c.name for c in app.registered_commands} == {
        "version",
        "doctor",
        "results",
        "compare",
        "run",
        "evaluate",
        "benchmark",
    }


def test_no_args_shows_help():
    result = runner.invoke(app, [])
    assert "Usage" in result.output


def test_version_output():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"Agent Harness Lab {__version__}"
    assert __version__ == "0.1.0"


def test_version_has_one_source_and_matches_package_metadata():
    import importlib.metadata as metadata
    import re
    from pathlib import Path

    pyproject = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text()
    assert 'dynamic = ["version"]' in pyproject  # no second, hand-written version string
    assert not re.search(r'^version\s*=', pyproject, re.MULTILINE)
    assert 'path = "src/agent_harness_lab/__init__.py"' in pyproject
    assert metadata.version("agent-harness-lab") == __version__


def test_doctor_renders_table_and_exits_zero(use_probe, healthy_probe):
    use_probe(healthy_probe)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Docker daemon" in result.output
    assert "OPENAI_API_KEY" in result.output
    assert "0 error(s)" in result.output
    assert "sk-super-secret-value" not in result.output


def test_doctor_with_only_optional_missing_exits_zero(use_probe, healthy_probe):
    for name in ("pi", "opencode", "node"):
        del healthy_probe.executables[name]
    healthy_probe.env.clear()
    use_probe(healthy_probe)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "WARNING" in result.output
    assert "not installed" in result.output


def test_doctor_foundational_error_exits_nonzero(use_probe, healthy_probe):
    healthy_probe.py = (3, 10, 0)
    use_probe(healthy_probe)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "ERROR" in result.output
