"""ahl-experiment-v1: validation, normalization, identity, matrix expansion."""

from __future__ import annotations

import json
import random
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from agent_harness_lab.experiments.errors import ExperimentError
from agent_harness_lab.experiments.loader import (
    load_experiment,
    make_experiment_id,
    parse_experiment,
    read_task_set,
)
from agent_harness_lab.experiments.matrix import cell_id, expand_matrix
from tests.experiment_fixtures import experiment_yaml

REPO = Path(__file__).resolve().parent.parent


def parse(text: str, base: Path | None = None):
    return parse_experiment(text.encode(), base_dir=base or REPO, source=Path("e.yaml"))


def test_valid_experiment_loads_and_normalizes():
    loaded = parse(
        experiment_yaml(tasks=("b__b-2", "a__a-1"), harnesses=("pi", "opencode"), repetitions=2)
    )
    n = loaded.normalized
    assert n.schema_version == "ahl-experiment-v1"
    assert n.tasks == ["a__a-1", "b__b-2"]  # normalized: sorted
    assert n.harnesses == ["opencode", "pi"]
    assert n.models[0].ref == "openai/gpt-6-astra" and n.models[0].reasoning == "medium"
    assert n.execution.network.value == "restricted-provider"
    assert n.evaluation.enabled is True and n.repetitions == 2


def test_defaults_are_filled_in():
    text = (
        "schema_version: ahl-experiment-v1\nexperiment: {name: d}\ntasks: [a__a-1]\n"
        "harnesses: [pi]\nmodels: [{provider: openai, model: m}]\n"
    )
    n = parse(text).normalized
    assert n.repetitions == 1 and n.execution.timeout_seconds == 3600
    assert n.evaluation.enabled is True and n.benchmark.name == "swebench-multilingual"
    assert n.benchmark.task_repository_commit is None and n.models[0].reasoning is None


def test_malformed_yaml_is_a_clean_error():
    with pytest.raises(ExperimentError, match="not valid YAML"):
        parse("schema_version: [unclosed\n  - x: : :\n")


def test_non_mapping_top_level_is_rejected():
    with pytest.raises(ExperimentError, match="mapping"):
        parse("- just\n- a list\n")


@pytest.mark.parametrize("version", ["ahl-experiment-v2", "1", "null"])
def test_unsupported_schema_version(version):
    text = experiment_yaml().replace("ahl-experiment-v1", version, 1)
    with pytest.raises(ExperimentError, match="unsupported schema_version"):
        parse(text)


def test_missing_schema_version_is_unsupported():
    with pytest.raises(ExperimentError, match="unsupported schema_version"):
        parse("experiment: {name: x}\n")


def test_duplicate_tasks_are_rejected():
    with pytest.raises(ExperimentError, match="duplicate task"):
        parse(experiment_yaml(tasks=("a__a-1", "b__b-2", "a__a-1")))


def test_duplicate_harness_and_model_are_rejected():
    with pytest.raises(ExperimentError, match="duplicate harness"):
        parse(experiment_yaml(harnesses=("pi", "pi")))
    with pytest.raises(ExperimentError, match="duplicate model"):
        parse(experiment_yaml(models=(("openai", "m", "medium"), ("openai", "m", "medium"))))


def test_same_model_with_different_reasoning_is_not_a_duplicate():
    n = parse(experiment_yaml(models=(("openai", "m", "low"), ("openai", "m", "high")))).normalized
    assert [m.reasoning for m in n.models] == ["high", "low"]


def test_unknown_top_level_fields_are_rejected():
    with pytest.raises(ExperimentError, match="api_key: Extra inputs are not permitted"):
        parse(experiment_yaml() + "api_key: sk-abc\n")  # secrets have no place in this file


def test_repetitions_must_be_positive():
    with pytest.raises(ExperimentError, match="repetitions"):
        parse(experiment_yaml(repetitions=0))


def test_a_key_pasted_into_a_model_entry_is_rejected():
    text = experiment_yaml().replace(
        "    reasoning: medium", "    reasoning: medium\n    api_key: sk-1"
    )
    with pytest.raises(ExperimentError, match="api_key"):
        parse(text)


@pytest.mark.parametrize("bad", ["Medium", "med ium", "1x", "", "x/y"])
def test_invalid_reasoning_level(bad):
    with pytest.raises(ExperimentError, match="reasoning"):
        parse(experiment_yaml().replace("reasoning: medium", f"reasoning: '{bad}'"))


@pytest.mark.parametrize("bad", ["abc", "G" * 40, "a" * 39, "a" * 41])
def test_invalid_task_repository_commit(bad):
    with pytest.raises(ExperimentError, match="40-character"):
        parse(experiment_yaml(pin=bad))


def test_pin_is_normalized_to_lowercase():
    assert (
        parse(experiment_yaml(pin="A" * 40)).normalized.benchmark.task_repository_commit == "a" * 40
    )


def test_timeout_must_be_positive():
    with pytest.raises(ExperimentError, match="timeout_seconds"):
        parse(experiment_yaml(timeout=0))


def test_error_messages_never_echo_input_values():
    secret = "sk-this-must-not-be-echoed"
    with pytest.raises(ExperimentError) as exc:
        parse(experiment_yaml().replace("model: gpt-6-astra", f"model: ''\n    api_key: {secret}"))
    assert secret not in str(exc.value)


# ---- task-set files ------------------------------------------------------------------------
def test_task_set_file_challenge_set_v1():
    ids, digest = read_task_set(REPO / "configs" / "challenge-set-v1.json")
    assert len(ids) == 6 and "babel__babel-15445" in ids and "axios__axios-5892" not in ids
    assert len(digest) == 64


def test_tasks_from_a_set_file_resolve_relative_to_the_experiment(tmp_path):
    (tmp_path / "set.json").write_text(json.dumps({"tasks": ["b__b-2", "a__a-1"]}))
    text = experiment_yaml().replace(
        "tasks:\n  - babel__babel-15445\n", "tasks:\n  file: set.json\n"
    )
    loaded = parse(text, tmp_path)
    assert loaded.normalized.tasks == ["a__a-1", "b__b-2"]
    assert loaded.normalized.task_set and loaded.normalized.task_set.file == "set.json"


def test_task_set_content_is_pinned_by_hash(tmp_path):
    text = experiment_yaml().replace(
        "tasks:\n  - babel__babel-15445\n", "tasks:\n  file: set.json\n"
    )
    (tmp_path / "set.json").write_text(json.dumps({"tasks": ["a__a-1"]}))
    first = parse(text, tmp_path)
    (tmp_path / "set.json").write_text(json.dumps({"tasks": ["a__a-1", "b__b-2"]}))
    assert parse(text, tmp_path).sha256 != first.sha256


def test_duplicates_inside_a_task_set_are_rejected(tmp_path):
    (tmp_path / "set.json").write_text(json.dumps({"tasks": ["a__a-1", "a__a-1"]}))
    text = experiment_yaml().replace(
        "tasks:\n  - babel__babel-15445\n", "tasks:\n  file: set.json\n"
    )
    with pytest.raises(ExperimentError, match="duplicate task"):
        parse(text, tmp_path)


@pytest.mark.parametrize("content", ["{}", "[]", '{"tasks": []}', '{"tasks": [1]}', "not: [valid"])
def test_bad_task_sets(tmp_path, content):
    (tmp_path / "set.json").write_text(content)
    with pytest.raises(ExperimentError):
        read_task_set(tmp_path / "set.json")


def test_missing_task_set_file(tmp_path):
    with pytest.raises(ExperimentError, match="cannot read task set"):
        read_task_set(tmp_path / "nope.json")


# ---- identity -----------------------------------------------------------------------------
def test_experiment_hash_is_stable_and_sensitive():
    base = parse(experiment_yaml()).sha256
    assert base == parse(experiment_yaml()).sha256 and re.fullmatch(r"[0-9a-f]{64}", base)
    assert parse(experiment_yaml(repetitions=2)).sha256 != base
    assert parse(experiment_yaml(evaluation=False)).sha256 != base
    assert parse(experiment_yaml(tasks=("babel__babel-15446",))).sha256 != base
    assert parse(experiment_yaml(timeout=10)).sha256 != base


def test_hash_ignores_comments_whitespace_and_list_order():
    a = parse(experiment_yaml(tasks=("a__a-1", "b__b-2"), harnesses=("pi", "opencode")))
    reordered = experiment_yaml(tasks=("b__b-2", "a__a-1"), harnesses=("opencode", "pi"))
    b = parse("# a comment\n\n" + reordered)
    assert a.sha256 == b.sha256


def test_hash_is_of_the_normalized_definition_not_of_the_raw_bytes():
    loaded = parse(experiment_yaml())
    import hashlib

    assert loaded.sha256 == hashlib.sha256(loaded.normalized.canonical_json().encode()).hexdigest()
    assert loaded.raw_bytes == experiment_yaml().encode()


def test_experiment_id_format():
    when = datetime(2026, 11, 24, 18, 0, 0, tzinfo=UTC)
    eid = make_experiment_id("Astra Comparison!", when, "a31c82f4" + "0" * 56)
    assert eid == "astra-comparison-20261124T180000Z-a31c82f4"
    assert make_experiment_id("***", when, "ab" * 32).startswith("experiment-2026")


def test_load_experiment_reads_a_file(tmp_path):
    p = tmp_path / "e.yaml"
    p.write_text(experiment_yaml())
    assert load_experiment(p).normalized.experiment.name == "astra-test"
    with pytest.raises(ExperimentError, match="cannot read experiment file"):
        load_experiment(tmp_path / "missing.yaml")


# ---- matrix --------------------------------------------------------------------------------
def test_matrix_size_is_the_product():
    n = parse(
        experiment_yaml(
            tasks=("a__a-1", "b__b-2", "c__c-3"),
            harnesses=("pi", "opencode"),
            models=(("openai", "m1", "low"), ("openai", "m2", None)),
            repetitions=2,
        )
    ).normalized
    assert len(expand_matrix(n)) == 3 * 2 * 2 * 2


def test_matrix_order_and_ids_are_pinned():
    n = parse(
        experiment_yaml(
            tasks=("b__b-2", "a__a-1"),
            harnesses=("pi", "opencode"),
            models=(("openai", "m", "medium"),),
            repetitions=2,
        )
    ).normalized
    assert [c.cell_id for c in expand_matrix(n)] == [
        "task=a__a-1;harness=opencode;model=openai/m;reasoning=medium;repetition=1",
        "task=a__a-1;harness=opencode;model=openai/m;reasoning=medium;repetition=2",
        "task=a__a-1;harness=pi;model=openai/m;reasoning=medium;repetition=1",
        "task=a__a-1;harness=pi;model=openai/m;reasoning=medium;repetition=2",
        "task=b__b-2;harness=opencode;model=openai/m;reasoning=medium;repetition=1",
        "task=b__b-2;harness=opencode;model=openai/m;reasoning=medium;repetition=2",
        "task=b__b-2;harness=pi;model=openai/m;reasoning=medium;repetition=1",
        "task=b__b-2;harness=pi;model=openai/m;reasoning=medium;repetition=2",
    ]


def test_matrix_is_independent_of_yaml_order():
    tasks = ["t__a-1", "t__b-2", "t__c-3", "t__d-4"]
    harnesses = ["pi", "opencode"]
    models = [("openai", "m1", "low"), ("openai", "m1", "high"), ("openai", "m0", None)]
    reference = [
        c.cell_id
        for c in expand_matrix(
            parse(experiment_yaml(tasks=tasks, harnesses=harnesses, models=models)).normalized
        )
    ]
    rng = random.Random(7)
    for _ in range(10):
        rng.shuffle(tasks), rng.shuffle(harnesses), rng.shuffle(models)
        got = [
            c.cell_id
            for c in expand_matrix(
                parse(experiment_yaml(tasks=tasks, harnesses=harnesses, models=models)).normalized
            )
        ]
        assert got == reference


def test_cells_without_reasoning_have_an_explicit_id():
    assert cell_id("t", "pi", "p", "m", None, 1).endswith("reasoning=none;repetition=1")


def test_cell_ids_are_unique_and_repetitions_are_one_based():
    n = parse(experiment_yaml(repetitions=3)).normalized
    cells = expand_matrix(n)
    assert len({c.cell_id for c in cells}) == 3 and [c.repetition for c in cells] == [1, 2, 3]


def test_model_ids_containing_slashes_are_kept_whole():
    n = parse(experiment_yaml(models=(("bme-lab", "Qwen/Qwen3-35B", None),))).normalized
    (c,) = expand_matrix(n)
    assert c.model == "Qwen/Qwen3-35B" and c.model_ref == "bme-lab/Qwen/Qwen3-35B"


def test_examples_are_valid_public_and_secret_free():
    minimal = load_experiment(REPO / "examples" / "minimal.yaml")
    astra = load_experiment(REPO / "examples" / "astra-comparison.yaml")
    assert len(expand_matrix(minimal.normalized)) == 1
    assert len(astra.normalized.tasks) == 11 and len(expand_matrix(astra.normalized)) == 22
    assert astra.normalized.benchmark.task_repository_commit == (
        "6e08cbcd763c869abd7aae4bcb9e1b3991180777"
    )
    for path in (REPO / "examples").glob("*.yaml"):
        text = path.read_text().lower()
        assert "bme" not in text and "sk-" not in text and "http" not in text
        assert yaml.safe_load(text)["schema_version"] == "ahl-experiment-v1"
    assert "not the original study" in (REPO / "examples" / "astra-comparison.yaml").read_text()
