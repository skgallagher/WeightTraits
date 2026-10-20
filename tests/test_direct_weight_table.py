from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Callable

import pytest
import yaml

from weighttraits.paper.direct_weight_table import (
    ALL_TREE_IDS,
    DIRECT_WEIGHT_TABLE_SCHEMA,
    TOPOLOGY_EXCLUDED_TREE_IDS,
    TOPOLOGY_TREE_IDS,
    build_direct_weight_table,
)


ORDERING_IDS = frozenset(
    {
        "confirm_paper_tree_003",
        "confirm_paper_tree_004",
        "confirm_paper_tree_006",
        "confirm_paper_tree_007",
        "confirm_paper_tree_009",
        "confirm_paper_tree_010",
        "confirm_paper_tree_011",
        "confirm_paper_tree_013",
        "confirm_paper_tree_016",
        "confirm_paper_tree_017",
        "confirm_paper_tree_019",
        "confirm_paper_tree_021",
        "confirm_paper_tree_023",
        "confirm_paper_tree_024",
        "confirm_paper_tree_025",
        "confirm_paper_tree_026",
        "confirm_paper_tree_027",
        "confirm_paper_tree_034",
        "confirm_paper_tree_035",
        "confirm_paper_tree_038",
        "confirm_paper_tree_039",
        "confirm_paper_tree_041",
        "confirm_paper_tree_042",
        "confirm_paper_tree_044",
        "confirm_paper_tree_046",
        "confirm_paper_tree_050",
    }
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _strict_completion_receipt(cohort_id: str) -> dict[str, Any]:
    rows = []
    remaining = 641
    for index, tree_id in enumerate(ALL_TREE_IDS):
        n_runs = 12 if index < 9 else 13
        remaining -= n_runs
        rows.append(
            {
                "tree_id": tree_id,
                "valid": True,
                "ready_for_analysis": True,
                "n_runs": n_runs,
                "n_ledger_nodes": n_runs,
                "n_terminal_nodes": n_runs,
                "n_ok_nodes": n_runs,
                "n_failed_nodes": 0,
                "n_missing_nodes": 0,
                "n_errors": 0,
                "n_expected_artifacts": 2 * n_runs,
                "n_existing_artifacts": 2 * n_runs,
                "status_counts": {"completed": n_runs},
            }
        )
    assert remaining == 0
    return {
        "cohort_id": cohort_id,
        "valid": True,
        "n_trees": 50,
        "n_ready": 50,
        "n_failed": 0,
        "n_in_progress": 0,
        "n_not_started": 0,
        "n_total_runs": 641,
        "n_terminal_nodes": 641,
        "n_ok_nodes": 641,
        "n_failed_nodes": 0,
        "n_missing_nodes": 0,
        "n_errors": 0,
        "n_warnings": 0,
        "rows": rows,
    }


def _make_case(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    truth_specs: dict[str, dict[str, str]] = {}
    truth_paths: dict[str, Path] = {}
    for tree_id in ALL_TREE_IDS:
        path = tmp_path / f"{tree_id}.manifest.jsonl"
        path.write_text(json.dumps({"tree_id": tree_id, "node_id": "n0"}) + "\n")
        truth_paths[tree_id] = path
        if tree_id not in TOPOLOGY_EXCLUDED_TREE_IDS:
            truth_specs[tree_id] = {"path": str(path), "sha256": _sha256(path)}

    conditions = []
    for condition_index in range(2):
        cohort_id = f"condition_{condition_index + 1}"
        contract = tmp_path / f"{cohort_id}.contract.json"
        contract_rows = []
        remaining = 641
        for index, tree_id in enumerate(ALL_TREE_IDS):
            n_runs = 12 if index < 9 else 13
            remaining -= n_runs
            contract_rows.append(
                {
                    "tree_id": tree_id,
                    "valid": True,
                    "n_runs": n_runs,
                    "n_errors": 0,
                    "n_warnings": 0,
                    "manifest_sha256": _sha256(truth_paths[tree_id]),
                }
            )
        assert remaining == 0
        _write_json(
            contract,
            {
                "valid": True,
                "config": f"examples/training/{cohort_id}.yaml",
                "n_trees": 50,
                "n_runs": 641,
                "n_errors": 0,
                "n_warnings": 0,
                "trees": contract_rows,
            },
        )
        completion = tmp_path / f"{cohort_id}.completion.json"
        _write_json(completion, _strict_completion_receipt(cohort_id))
        rollup = tmp_path / f"{cohort_id}.rollup.json"
        rows = []
        eligible_index = 0
        ordering_index = 0
        for tree_id in ALL_TREE_IDS:
            excluded = tree_id in TOPOLOGY_EXCLUDED_TREE_IDS
            ordering_valid = not excluded and tree_id in ORDERING_IDS
            row = {
                "tree_id": tree_id,
                "metric": "cosine",
                "artifact": "model",
                "representation": "full_weight",
                "n_truth_splits": 0 if excluded else 1,
                "truth_manifest": str(truth_paths[tree_id]),
                "branch_ordering_valid": ordering_valid,
                "branch_ordering_status": "ok" if ordering_valid else "missing_branch_class",
                "branch_rank_biserial": (
                    -0.5 + 0.04 * ordering_index if ordering_valid else None
                ),
                "branch_within_run_r": (
                    -0.6 + 0.04 * ordering_index if ordering_valid else None
                ),
                "clade_recovery": 0.5 if excluded else eligible_index / 100.0,
                "polytomy_aware_exact_recovery": (
                    False if excluded else eligible_index % 2 == 0
                ),
                "rf": 0 if excluded else eligible_index,
                "false_negative": 0 if excluded else eligible_index % 3,
            }
            rows.append(row)
            if not excluded:
                eligible_index += 1
            if ordering_valid:
                ordering_index += 1
        _write_json(
            rollup,
            {
                "valid": True,
                "artifact": "model",
                "rows": rows,
                "aggregate_by_metric": {
                    "cosine": {
                        # Deliberately wrong: the aggregator must ignore this block.
                        "clade_recovery_mean": 999,
                        "n_ordering_trees": 999,
                    }
                },
            },
        )
        conditions.append(
            {
                "cohort_id": cohort_id,
                "architecture": "Synthetic",
                "training": f"method {condition_index + 1}",
                "artifact": "model",
                "representation": "full_weight",
                "cohort_contract": {
                    "path": str(contract),
                    "sha256": _sha256(contract),
                },
                "completion_receipt": {
                    "path": str(completion),
                    "sha256": _sha256(completion),
                },
                "rollup": {"path": str(rollup), "sha256": _sha256(rollup)},
            }
        )

    config = {
        "version": 1,
        "metric": "cosine",
        "required_cohort_ids": [row["cohort_id"] for row in conditions],
        "ordering_tree_ids": sorted(ORDERING_IDS),
        "truth_manifests": truth_specs,
        "conditions": conditions,
    }
    config_path = tmp_path / "direct_weight_table.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path, config


def _mutate_rollup(
    config_path: Path,
    condition_index: int,
    mutator: Callable[[dict[str, Any]], None],
) -> None:
    config = yaml.safe_load(config_path.read_text())
    source_spec = config["conditions"][condition_index]["rollup"]
    source = Path(source_spec["path"])
    payload = json.loads(source.read_text())
    mutator(payload)
    _write_json(source, payload)
    source_spec["sha256"] = _sha256(source)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))


def test_build_direct_weight_table_recomputes_required_estimators(tmp_path: Path):
    config_path, _ = _make_case(tmp_path)

    payload = build_direct_weight_table(config_path)

    assert payload["schema"] == DIRECT_WEIGHT_TABLE_SCHEMA
    assert payload["topology_tree_ids"] == list(TOPOLOGY_TREE_IDS)
    assert payload["ordering_tree_ids"] == sorted(ORDERING_IDS)
    assert payload["n_rows"] == 2
    row = payload["rows"][0]
    topology_values = list(range(46))
    rank_values = [-0.5 + 0.04 * index for index in range(26)]
    raw_r_values = [-0.6 + 0.04 * index for index in range(26)]
    expected_paer = 0.5

    assert row["n_recovery"] == 46
    assert row["n_ordering"] == 26
    assert row["clade_recovery"] == pytest.approx(
        statistics.fmean(value / 100.0 for value in topology_values)
    )
    assert row["clade_recovery_se"] == pytest.approx(
        statistics.stdev(value / 100.0 for value in topology_values) / math.sqrt(46)
    )
    assert row["paer"] == expected_paer
    assert row["paer_se"] == pytest.approx(
        math.sqrt(expected_paer * (1.0 - expected_paer) / 46)
    )
    assert row["rf"] == statistics.fmean(topology_values)
    assert row["rf_se"] == pytest.approx(
        statistics.stdev(topology_values) / math.sqrt(46)
    )
    assert row["false_negative"] == pytest.approx(
        statistics.fmean(value % 3 for value in topology_values)
    )
    assert row["false_negative_se"] == pytest.approx(
        statistics.stdev(value % 3 for value in topology_values) / math.sqrt(46)
    )
    assert row["rank_biserial"] == pytest.approx(statistics.fmean(rank_values))
    assert row["rank_biserial_se"] == pytest.approx(
        statistics.stdev(rank_values) / math.sqrt(26)
    )
    assert row["within_run_r"] == pytest.approx(
        math.tanh(statistics.fmean(math.atanh(value) for value in raw_r_values))
    )
    assert row["within_run_r_se"] == pytest.approx(
        statistics.stdev(raw_r_values) / math.sqrt(26)
    )
    assert row["clade_recovery"] != 999


@pytest.mark.parametrize(
    "target", ["cohort_contract", "completion_receipt", "rollup"]
)
def test_build_direct_weight_table_rejects_unpinned_input_changes(
    tmp_path: Path,
    target: str,
):
    config_path, config = _make_case(tmp_path)
    path = Path(config["conditions"][0][target]["path"])
    path.write_text(path.read_text() + " \n")

    with pytest.raises(ValueError, match="sha256 mismatch"):
        build_direct_weight_table(config_path)


def test_build_direct_weight_table_rejects_declaration_only_completion_receipt(
    tmp_path: Path,
) -> None:
    config_path, config = _make_case(tmp_path)
    spec = config["conditions"][0]["completion_receipt"]
    receipt = Path(spec["path"])
    _write_json(receipt, {"cohort_id": "condition_1", "valid": True, "n_trees": 50})
    loaded = yaml.safe_load(config_path.read_text())
    loaded["conditions"][0]["completion_receipt"]["sha256"] = _sha256(receipt)
    config_path.write_text(yaml.safe_dump(loaded, sort_keys=False))

    with pytest.raises(ValueError, match="strict training-completion"):
        build_direct_weight_table(config_path)


def test_build_direct_weight_table_rejects_truth_hash_mismatch(tmp_path: Path):
    config_path, config = _make_case(tmp_path)
    truth_path = Path(config["truth_manifests"][TOPOLOGY_TREE_IDS[0]]["path"])
    truth_path.write_text(truth_path.read_text() + " \n")

    with pytest.raises(ValueError, match="truth manifest.*sha256 mismatch"):
        build_direct_weight_table(config_path)


def test_build_direct_weight_table_rejects_duplicate_tree_rows(tmp_path: Path):
    config_path, _ = _make_case(tmp_path)

    def duplicate(payload: dict[str, Any]) -> None:
        payload["rows"].append(dict(payload["rows"][0]))

    _mutate_rollup(config_path, 0, duplicate)
    with pytest.raises(ValueError, match="duplicate 'cosine' row"):
        build_direct_weight_table(config_path)


def test_build_direct_weight_table_rejects_wrong_topology_id_set(tmp_path: Path):
    config_path, _ = _make_case(tmp_path)

    def remove_tree(payload: dict[str, Any]) -> None:
        payload["rows"] = [
            row for row in payload["rows"] if row["tree_id"] != TOPOLOGY_TREE_IDS[0]
        ]

    _mutate_rollup(config_path, 0, remove_tree)
    with pytest.raises(ValueError, match="exactly one 'cosine' row for each of 50 trees"):
        build_direct_weight_table(config_path)


def test_build_direct_weight_table_rejects_missing_required_condition(
    tmp_path: Path,
) -> None:
    config_path, _ = _make_case(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    config["conditions"] = config["conditions"][:1]
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    with pytest.raises(ValueError, match="required_cohort_ids"):
        build_direct_weight_table(config_path)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("artifact", "merged", "artifact mismatch"),
        ("representation", "adapter_delta", "representation mismatch"),
    ],
)
def test_build_direct_weight_table_rejects_row_provenance_mismatch(
    tmp_path: Path,
    field: str,
    replacement: str,
    message: str,
):
    config_path, _ = _make_case(tmp_path)

    def change(payload: dict[str, Any]) -> None:
        payload["rows"][0][field] = replacement

    _mutate_rollup(config_path, 0, change)
    with pytest.raises(ValueError, match=message):
        build_direct_weight_table(config_path)


def test_build_direct_weight_table_rejects_different_ordering_sets(tmp_path: Path):
    config_path, _ = _make_case(tmp_path)
    old_id = sorted(ORDERING_IDS)[0]
    new_id = next(tree_id for tree_id in TOPOLOGY_TREE_IDS if tree_id not in ORDERING_IDS)

    def swap(payload: dict[str, Any]) -> None:
        rows = {row["tree_id"]: row for row in payload["rows"]}
        old = rows[old_id]
        new = rows[new_id]
        old["branch_ordering_valid"] = False
        old["branch_ordering_status"] = "missing_branch_class"
        old["branch_rank_biserial"] = None
        old["branch_within_run_r"] = None
        new["branch_ordering_valid"] = True
        new["branch_ordering_status"] = "ok"
        new["branch_rank_biserial"] = 0.2
        new["branch_within_run_r"] = -0.2

    _mutate_rollup(config_path, 1, swap)
    with pytest.raises(ValueError, match="ordering-valid tree IDs differ from declared"):
        build_direct_weight_table(config_path)


def test_build_direct_weight_table_requires_exact_truth_declarations(tmp_path: Path):
    config_path, config = _make_case(tmp_path)
    config["truth_manifests"].pop(TOPOLOGY_TREE_IDS[0])
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    with pytest.raises(ValueError, match="exactly the 46 topology trees"):
        build_direct_weight_table(config_path)


def test_build_direct_weight_table_requires_declared_exact_ordering_set(tmp_path: Path):
    config_path, config = _make_case(tmp_path)
    config["ordering_tree_ids"].pop()
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    with pytest.raises(ValueError, match="exactly 26 trees"):
        build_direct_weight_table(config_path)


def test_build_direct_weight_table_rejects_identical_truth_at_other_path(
    tmp_path: Path,
):
    config_path, config = _make_case(tmp_path)
    tree_id = TOPOLOGY_TREE_IDS[0]
    original = Path(config["truth_manifests"][tree_id]["path"])
    alternate = tmp_path / "alternate.manifest.jsonl"
    alternate.write_bytes(original.read_bytes())

    def change_truth_path(payload: dict[str, Any]) -> None:
        for row in payload["rows"]:
            if row["tree_id"] == tree_id:
                row["truth_manifest"] = str(alternate)

    _mutate_rollup(config_path, 0, change_truth_path)
    with pytest.raises(ValueError, match="truth_manifest path mismatch"):
        build_direct_weight_table(config_path)


def test_build_direct_weight_table_rejects_invalid_completion_receipt(tmp_path: Path):
    config_path, config = _make_case(tmp_path)
    receipt_spec = config["conditions"][0]["completion_receipt"]
    receipt = Path(receipt_spec["path"])
    _write_json(receipt, {"cohort_id": "condition_1", "valid": False})
    current = yaml.safe_load(config_path.read_text())
    current["conditions"][0]["completion_receipt"]["sha256"] = _sha256(receipt)
    config_path.write_text(yaml.safe_dump(current, sort_keys=False))

    with pytest.raises(ValueError, match="strict training-completion"):
        build_direct_weight_table(config_path)
