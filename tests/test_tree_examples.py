import json
from pathlib import Path

import yaml

from weighttraits.trees.generate import generate_tree_from_config, manifest_rows, tree_stats


EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "trees"


def _example_paths():
    return sorted(path for path in EXAMPLE_DIR.glob("*.yaml"))


def _check_expectations(stats: dict, expect: dict) -> None:
    exact_keys = ("n_nodes", "n_leaves", "max_depth", "max_out_degree", "n_polytomies")
    for key in exact_keys:
        if key in expect:
            assert stats[key] == expect[key], f"{key}: got {stats[key]}, expected {expect[key]}"

    min_map = {
        "min_nodes": "n_nodes",
        "min_leaves": "n_leaves",
        "min_depth": "max_depth",
        "min_max_out_degree": "max_out_degree",
        "min_polytomies": "n_polytomies",
    }
    for expect_key, stats_key in min_map.items():
        if expect_key in expect:
            assert stats[stats_key] >= expect[expect_key], (
                f"{stats_key}: got {stats[stats_key]}, expected >= {expect[expect_key]}"
            )

    max_map = {
        "max_nodes": "n_nodes",
        "max_leaves": "n_leaves",
        "max_depth": "max_depth",
        "max_max_out_degree": "max_out_degree",
        "max_polytomies": "n_polytomies",
    }
    for expect_key, stats_key in max_map.items():
        if expect_key in expect and expect_key not in exact_keys:
            assert stats[stats_key] <= expect[expect_key], (
                f"{stats_key}: got {stats[stats_key]}, expected <= {expect[expect_key]}"
            )


def test_all_tree_examples_have_expectations():
    paths = _example_paths()
    assert paths, "no tree examples found"
    for path in paths:
        config = yaml.safe_load(path.read_text())
        assert "tree" in config, path
        assert "expect" in config, path


def test_all_tree_examples_generate_valid_manifests():
    for path in _example_paths():
        config = yaml.safe_load(path.read_text())
        root = generate_tree_from_config(config["tree"])
        rows = manifest_rows(root)
        stats = tree_stats(root)

        assert len(rows) == stats["n_nodes"], path
        assert all(row["node_id"] != "root" for row in rows), path
        assert all(row["path"][0] == "root" for row in rows), path
        _check_expectations(stats, config["expect"])


def test_confirm_paper_numbers_tree_set_has_50_min_leaf_manifests():
    repo = Path(__file__).resolve().parents[1]
    summary_path = repo / "examples/training/confirm_paper_numbers/tree_set_summary.json"
    summary = json.loads(summary_path.read_text())

    assert summary["n_trees"] == 50
    assert summary["min_leaves"] == 4
    assert len(summary["trees"]) == 50
    for tree in summary["trees"]:
        manifest = repo / tree["manifest"]
        rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        leaf_ids = {row["node_id"] for row in rows} - {row["parent_id"] for row in rows if row["parent_id"]}
        assert len(rows) == tree["n_rows"]
        assert len(leaf_ids) == tree["n_leaves"]
        assert tree["n_leaves"] >= 4


def test_confirm_paper_numbers_assignments_match_tree_set_without_replacement():
    repo = Path(__file__).resolve().parents[1]
    config_path = repo / "examples/training/confirm_paper_numbers/paper_task_families.yaml"
    summary_path = repo / "examples/training/confirm_paper_numbers/assignment_summary.json"
    config = yaml.safe_load(config_path.read_text())
    summary = json.loads(summary_path.read_text())

    family_sizes = {
        family: len(spec["datasets"])
        for family, spec in config["task_families"].items()
    }
    assert family_sizes == {
        "classification": 10,
        "qa": 8,
        "summarization": 9,
        "translation": 9,
    }
    assert summary["n_trees"] == 50
    assert summary["policy"] == "per_node_without_replacement"

    for assignment in summary["assignments"]:
        manifest = repo / assignment["assigned_manifest"]
        rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        train_rows = [row for row in rows if row.get("grow", "train") == "train"]
        pairs = [(row["task_family"], row["dataset_id"]) for row in train_rows]

        assert assignment["n_rows"] == len(rows)
        assert assignment["n_datasets"] == len(train_rows)
        assert assignment["n_unique_datasets"] == len(set(pairs))
        assert len(set(pairs)) == len(pairs)
        assert {row["tree_id"] for row in rows} == {assignment["tree_id"]}
