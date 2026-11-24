import json
import pathlib

import pytest
from typer.testing import CliRunner

from agent_harness_lab import paths
from agent_harness_lab.benchmarks.base import Task
from agent_harness_lab.benchmarks.selection import RULE, select_candidates
from agent_harness_lab.cli import common
from agent_harness_lab.cli import tasks as tasks_cli
from agent_harness_lab.cli.app import app
from agent_harness_lab.results.history import frozen_task_ids, task_history
from tests.benchmark_fixtures import SENTINELS, FakeGit, make_task_repo
from tests.factories import make_run
from tests.fake_docker import FakeDocker

runner = CliRunner()
WIDE = {"COLUMNS": "200"}


def task(i, image=None):
    return Task(
        benchmark="b", instance_id=i, repo="o/r", base_commit="a" * 40,
        docker_image=image or f"img/{i}:latest", problem_statement="s",
        task_directory=pathlib.Path("/x"), task_repository_commit=None,
    )  # fmt: skip


# ---- history ---------------------------------------------------------------------------------
def test_task_with_run_directories_is_seen(tmp_path):
    make_run(tmp_path, "r1", "pi-swebench-reference", instanceId="proj__a-1")
    h = task_history(tmp_path, "proj__a-1")
    assert h.seen and h.run_directories == 1 and h.harnesses == ["Pi"]
    assert h.models == ["gpt-6-astra"] and not h.frozen_research_task


def test_task_without_run_directories_is_unseen(tmp_path):
    (tmp_path / "swebench").mkdir()
    h = task_history(tmp_path, "proj__never-9")
    assert not h.seen and h.run_directories == 0 and h.harnesses == []


def test_incomplete_run_directory_still_counts_as_history(tmp_path):
    (tmp_path / "swebench" / "proj__a-1" / "proj__a-1-aborted").mkdir(parents=True)
    h = task_history(tmp_path, "proj__a-1")
    assert h.seen and h.run_directories == 1 and h.harnesses == []


def test_frozen_research_task_is_seen_even_without_runs(tmp_path):
    (tmp_path / "swebench").mkdir()
    h = task_history(tmp_path, "proj__a-1", frozen={"proj__a-1"})
    assert h.seen and h.frozen_research_task and h.run_directories == 0


def test_history_includes_opencode_and_canary_purposes(tmp_path):
    make_run(tmp_path, "r1", "opencode-swebench", instanceId="proj__a-1")
    assert task_history(tmp_path, "proj__a-1").harnesses == ["OpenCode"]


def test_history_never_reads_outcomes(tmp_path, monkeypatch):
    make_run(tmp_path, "r1", "pi-swebench-reference", instanceId="proj__a-1", outcome="resolved")
    opened = []
    real = pathlib.Path.open

    def spy(self, *a, **k):
        opened.append(self.name)
        return real(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "open", spy)
    task_history(tmp_path, "proj__a-1")
    assert opened and "evaluation.json" not in opened  # outcome-independent by construction


def test_frozen_ids_come_from_the_committed_manifests(tmp_path):
    (tmp_path / "experiments/swebench").mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    (tmp_path / "experiments/swebench/swebench-pilot-5-v1.json").write_text(
        json.dumps({"instances": [{"instanceId": "a__a-1"}, {"instanceId": "b__b-2"}]})
    )
    (tmp_path / "configs/challenge-set-v1.json").write_text(
        json.dumps({"ecosystems": {"x": ["c__c-3"], "y": ["d__d-4"]}})
    )
    assert frozen_task_ids(tmp_path) == {"a__a-1", "b__b-2", "c__c-3", "d__d-4"}
    assert frozen_task_ids(None) == set() and frozen_task_ids(tmp_path / "missing") == set()


# ---- deterministic selection -----------------------------------------------------------------
def test_selection_is_lexicographic_and_independent_of_input_order():
    tasks = [task(i) for i in ("c__c-3", "a__a-1", "d__d-4", "b__b-2")]
    pick = lambda ts: [  # noqa: E731
        c.instance_id
        for c in select_candidates(ts, is_seen=lambda i: False, image_local=lambda r: True).ready
    ]
    assert pick(tasks) == ["a__a-1", "b__b-2", "c__c-3"]
    assert pick(list(reversed(tasks))) == pick(tasks)


def test_selection_skips_seen_tasks_and_requires_a_local_image():
    tasks = [task(i) for i in ("a__a-1", "b__b-2", "c__c-3", "d__d-4")]
    sel = select_candidates(
        tasks, is_seen=lambda i: i == "a__a-1", image_local=lambda r: "c__c-3" not in r, count=2
    )
    assert [c.instance_id for c in sel.ready] == ["b__b-2", "d__d-4"]
    assert sel.unseen_total == 3 and sel.shortfall == 0 and sel.needs_image == []
    assert all(c.image_local for c in sel.ready) and sel.rule == RULE


def test_shortfall_is_reported_and_missing_images_are_flagged_not_hidden():
    tasks = [task(i) for i in ("a__a-1", "b__b-2", "c__c-3", "d__d-4")]
    sel = select_candidates(
        tasks, is_seen=lambda i: False, image_local=lambda r: "b__b-2" in r, count=3
    )
    assert [c.instance_id for c in sel.ready] == ["b__b-2"] and sel.shortfall == 2
    assert [(c.instance_id, c.image_local) for c in sel.needs_image] == [
        ("a__a-1", False),
        ("c__c-3", False),
    ]


# ---- CLI ---------------------------------------------------------------------------------------
@pytest.fixture
def cli_env(monkeypatch, tmp_path):
    repo = make_task_repo(tmp_path)  # babel__babel-15445, axios__axios-5892, apache__lucene-12212
    raw = tmp_path / "raw"
    (raw / "swebench" / "axios__axios-5892" / "old-run").mkdir(parents=True)
    docker = FakeDocker(
        images={
            "swebench/sweb.eval.x86_64.babel_1776_babel-15445:latest",
            "swebench/sweb.eval.x86_64.axios_1776_axios-5892:latest",
        }
    )
    monkeypatch.setattr(tasks_cli, "_git_probe", lambda: FakeGit())
    monkeypatch.setattr(common, "docker_client", lambda: docker)
    monkeypatch.setattr(paths, "find_project_root", lambda start=None: None)  # no manifests
    monkeypatch.setenv("AHL_SWEBENCH_TASKS_REPO", str(repo))
    return raw, docker


def test_history_cli_seen_and_unseen(cli_env):
    raw, _ = cli_env
    seen = runner.invoke(
        app, ["tasks", "history", "axios__axios-5892", "--raw-dir", str(raw)], env=WIDE
    )
    assert (
        seen.exit_code == 0 and "SEEN" in seen.output and "run directories:       1" in seen.output
    )
    unseen = runner.invoke(
        app, ["tasks", "history", "babel__babel-15445", "--raw-dir", str(raw), "--json"], env=WIDE
    )
    assert json.loads(unseen.stdout)["seen"] is False


def test_history_cli_without_a_result_store_refuses_to_claim_unseen(tmp_path, cli_env):
    r = runner.invoke(app, ["tasks", "history", "x", "--raw-dir", str(tmp_path / "none")], env=WIDE)
    assert r.exit_code == 1 and "history cannot be established" in r.output


def test_candidates_cli_selects_unseen_tasks_with_local_images(cli_env):
    raw, docker = cli_env
    r = runner.invoke(app, ["tasks", "candidates", "--raw-dir", str(raw), "--json"], env=WIDE)
    sel = json.loads(r.stdout)
    assert [c["instance_id"] for c in sel["ready"]] == [
        "babel__babel-15445"
    ]  # axios seen, lucene no image
    assert [c["instance_id"] for c in sel["needs_image"]] == ["apache__lucene-12212"]
    assert sel["requested"] == 3 and sel["rule"] == RULE
    assert docker.ops() == ["list_image_refs"]  # a single read-only listing; nothing executed


def test_candidates_human_output_reports_the_shortfall(cli_env):
    raw, _ = cli_env
    r = runner.invoke(app, ["tasks", "candidates", "--raw-dir", str(raw)], env=WIDE)
    assert "Rule:" in r.output and "Shortfall: 2" in r.output and "NO" in r.output


def test_candidate_selection_never_reads_gold_or_evaluator_files(cli_env, monkeypatch):
    raw, _ = cli_env
    opened = []
    real = pathlib.Path.open

    def spy(self, *a, **k):
        opened.append(self.name)
        return real(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "open", spy)
    r = runner.invoke(app, ["tasks", "candidates", "--raw-dir", str(raw)], env=WIDE)
    assert r.exit_code == 0
    assert set(opened) <= {"task.yaml", "problem_statement.md"}  # no gold/hints/tests/eval files
    for sentinel in SENTINELS:
        assert sentinel not in r.output

