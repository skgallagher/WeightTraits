import csv
import json
from pathlib import Path
import subprocess
import sys

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


def _write_ellmtrees_manifest(run_dir: Path, rows: list[dict]) -> None:
    run_dir.mkdir(parents=True)
    run_dir.joinpath("manifest.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
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


def test_confirm_paper_tree_comparison_report_smoke(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    ellmtrees_runs = tmp_path / "ellmtrees_runs"
    paper_tex = tmp_path / "paper.tex"
    out_dir = tmp_path / "report"

    paper_tex.write_text(
        r"""
Training trees are generated via a Poisson branching process.
$k \sim \mathrm{Poisson}(\lambda = 1.5)$.
The node budget is $n_\mathrm{nodes} = 14$ and the maximum depth is
$n_\mathrm{layers} = 4$. Runs with fewer than 4 leaves are excluded.
This procedure yields trees with 4--8 observed leaves across the 50 replications.
"""
    )

    _write_ellmtrees_manifest(
        ellmtrees_runs / "run_001",
        [
            {
                "node_id": "n0",
                "grow": "train",
                "depth": 1,
                "path": ["root", "n0"],
                "is_leaf": False,
            },
            {
                "node_id": "n1",
                "grow": "train",
                "depth": 2,
                "path": ["root", "n0", "n1"],
                "is_leaf": True,
            },
            {
                "node_id": "n2",
                "grow": "train",
                "depth": 2,
                "path": ["root", "n0", "n2"],
                "is_leaf": True,
            },
        ],
    )

    _write_ellmtrees_manifest(
        ellmtrees_runs / "run_002",
        [
            {
                "node_id": "n0",
                "grow": "train",
                "depth": 1,
                "path": ["root", "n0"],
                "is_leaf": False,
            },
            {
                "node_id": "n1",
                "grow": "train",
                "depth": 2,
                "path": ["root", "n0", "n1"],
                "is_leaf": True,
            },
            {
                "node_id": "n2",
                "grow": "train",
                "depth": 2,
                "path": ["root", "n0", "n2"],
                "is_leaf": True,
            },
            {
                "node_id": "n3",
                "grow": "train",
                "depth": 2,
                "path": ["root", "n0", "n3"],
                "is_leaf": True,
            },
        ],
    )

    subprocess.run(
        [
            sys.executable,
            str(repo / "scripts/compare_confirm_paper_trees.py"),
            "--weighttraits-summary",
            str(repo / "examples/training/confirm_paper_numbers/tree_set_summary.json"),
            "--ellmtrees-runs",
            str(ellmtrees_runs),
            "--paper-tex",
            str(paper_tex),
            "--out-dir",
            str(out_dir),
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )

    rows = list(csv.DictReader((out_dir / "source_summary.csv").open()))
    by_source = {row["source"]: row for row in rows}
    assert by_source["ELLMTrees runs_branching_v3"]["n_trees"] == "2"
    assert by_source["ELLMTrees runs_branching_v3"]["leaves_min"] == "2"
    assert by_source["WeightTraits confirm_paper_numbers"]["n_trees"] == "50"
    assert by_source["WeightTraits confirm_paper_numbers"]["leaves_max"] == "10"
    assert (out_dir / "README.md").exists()
    assert (out_dir / "leaf_count_distribution.svg").exists()
    assert (out_dir / "training_nodes_vs_leaves.svg").exists()


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


def test_confirm_paper_numbers_full_finetune_run_lists_match_assignments():
    repo = Path(__file__).resolve().parents[1]
    assignment_path = repo / "examples/training/confirm_paper_numbers/assignment_summary.json"
    run_summary_path = repo / "examples/training/confirm_paper_numbers/full_finetune_run_list_summary.json"
    assignment_summary = json.loads(assignment_path.read_text())
    run_summary = json.loads(run_summary_path.read_text())
    expected_runs = sum(assignment["n_rows"] for assignment in assignment_summary["assignments"])

    assert run_summary["valid"]
    assert run_summary["n_trees"] == 50
    assert run_summary["n_runs"] == expected_runs == 641
    assert run_summary["n_errors"] == 0
    assert run_summary["n_warnings"] == 0
    assert run_summary["runner_options"] == {"dry_run": True}

    run_list_paths = sorted(
        (repo / "examples/training/confirm_paper_numbers/full_finetune_runlists/run_lists").glob(
            "*.jsonl"
        )
    )
    assert len(run_list_paths) == 50
    for tree in run_summary["trees"]:
        run_list = repo / tree["run_list"]
        rows = [json.loads(line) for line in run_list.read_text().splitlines()]
        output_root = tree["output_root"]

        assert tree["valid"]
        assert len(rows) == tree["n_runs"]
        assert all(row["runner"]["options"] == {"dry_run": True} for row in rows)
        assert all(row["output_dir"].startswith(output_root) for row in rows)
        assert all(row["ledger_path"] == tree["ledger"] for row in rows)


def test_confirm_paper_numbers_training_run_lists_are_wired_for_runner():
    repo = Path(__file__).resolve().parents[1]
    summary_path = (
        repo / "examples/training/confirm_paper_numbers/full_finetune_training_run_list_summary.json"
    )
    summary = json.loads(summary_path.read_text())
    expected_options = {
        "allow_missing_eval": True,
        "data_cache_root": "data/confirm_paper_numbers/full_finetune_cache",
        "formats_path": "examples/training/confirm_paper_numbers/dataset_formats.yaml",
        "max_eval_samples": 1000,
        "max_train_samples": 10000,
        "registry_path": "examples/training/confirm_paper_numbers/dataset_registry.yaml",
        "require_data_cache": True,
    }

    assert summary["valid"]
    assert summary["n_trees"] == 50
    assert summary["n_runs"] == 641
    assert summary["n_errors"] == 0
    assert summary["n_warnings"] == 0
    assert summary["runner_entrypoint"] == "weighttraits.cli run-training-row"
    assert summary["runner_options"] == expected_options
    for tree in summary["trees"]:
        run_list = repo / tree["run_list"]
        rows = [json.loads(line) for line in run_list.read_text().splitlines()]
        assert len(rows) == tree["n_runs"]
        assert all(row["runner"]["options"] == expected_options for row in rows)


def test_confirm_paper_numbers_dataset_contract_reports_are_clean():
    repo = Path(__file__).resolve().parents[1]
    family_config = yaml.safe_load(
        (repo / "examples/training/confirm_paper_numbers/paper_task_families.yaml").read_text()
    )
    registry = yaml.safe_load(
        (repo / "examples/training/confirm_paper_numbers/dataset_registry.yaml").read_text()
    )
    formats = yaml.safe_load(
        (repo / "examples/training/confirm_paper_numbers/dataset_formats.yaml").read_text()
    )
    validation = json.loads(
        (repo / "examples/training/confirm_paper_numbers/full_finetune_data_format_validation.json").read_text()
    )
    no_load_audit = json.loads(
        (repo / "examples/training/confirm_paper_numbers/dataset_registry_no_load_audit.json").read_text()
    )
    cache_summary = json.loads(
        (repo / "examples/training/confirm_paper_numbers/full_finetune_data_cache_summary.json").read_text()
    )

    family_ids = {
        dataset["id"]
        for family in family_config["task_families"].values()
        for dataset in family["datasets"]
    }
    registry_ids = {
        dataset["id"]
        for family in registry["task_families"].values()
        for dataset in family["datasets"]
    }
    format_ids = {dataset["dataset_id"] for dataset in formats["datasets"]}

    assert len(family_ids) == 36
    assert registry_ids == family_ids
    assert format_ids == family_ids
    assert validation["valid"]
    assert validation["n_trees"] == 50
    assert validation["n_jobs"] == validation["n_valid"] == 641
    assert validation["n_issues"] == 0
    assert no_load_audit["valid"]
    assert no_load_audit["n_datasets"] == no_load_audit["n_ok"] == 36
    assert {audit["status"] for audit in no_load_audit["audits"]} == {"not_loaded"}
    assert cache_summary["valid"]
    assert cache_summary["n_datasets"] == cache_summary["n_ok"] == 36
    summarization_ids = {
        dataset["id"]
        for dataset in family_config["task_families"]["summarization"]["datasets"]
    }
    summarization_train = [
        split
        for split in cache_summary["splits"]
        if split["dataset_id"] in summarization_ids and split["split"] == "train"
    ]
    assert len(summarization_train) == 9
    assert all(split["n_cached"] == 10000 for split in summarization_train)
