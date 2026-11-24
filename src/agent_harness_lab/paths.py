"""Project/repository path discovery (no hardcoded home directories)."""

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path | None:
    """Nearest parent holding both pyproject.toml and package.json, but ONLY when the package
    itself lives in that checkout's ``src/`` (a source checkout / editable install).

    An installed wheel returns None even if its virtualenv happens to sit inside some other
    project that has those two files: that project is not AHL's, so its pins mean nothing here.
    """
    here = (start or Path(__file__)).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "package.json").is_file():
            return parent if (parent / "src") in here.parents else None
    return None


def default_raw_dir() -> Path:
    """Default location of raw run artifacts: <project-root>/results/raw, else ./results/raw."""
    root = find_project_root()
    return (root or Path.cwd()) / "results" / "raw"


def default_experiments_dir() -> Path:
    """Experiment artifacts (matrix, summary), kept apart from raw run directories."""
    root = find_project_root()
    return (root or Path.cwd()) / "results" / "experiments"
