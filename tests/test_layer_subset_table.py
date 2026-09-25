from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from weighttraits.paper.analysis_contracts import (
    ALL_TREE_IDS,
    TOPOLOGY_TREE_IDS,
    truth_hashes_sha256,
)
from weighttraits.paper.layer_subset_table import (
    DIRECT_TREE_ANALYSIS_SCHEMA,
    EXPECTED_SUBSET_COUNTS,
    LAYER_SUBSET_INVENTORY_SCHEMA,
    LAYER_SUBSET_TABLE_SCHEMA,
    build_layer_subset_inventory,
    build_layer_subset_table,
    resolve_layer_subsets,
    write_layer_subset_inventory_json,
)
from weighttraits.paper.direct_weight_table import DIRECT_WEIGHT_TABLE_SCHEMA


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _flan_layer_names() -> list[str]:
    names = []
    for stack in ("decoder", "encoder"):
        for block in range(12):
            prefix = f"{stack}.block.{block}"
            names.extend(
                f"{prefix}.layer.0.SelfAttention.{projection}.weight"
                for projection in ("k", "o", "q", "v")
            )
            if block == 0:
                names.append(f"{prefix}.layer.0.SelfAttention.relative_attention_bias.weight")
            names.append(f"{prefix}.layer.0.layer_norm.weight")
            if stack == "decoder":
                names.extend(
                    f"{prefix}.layer.1.EncDecAttention.{projection}.weight"
                    for projection in ("k", "o", "q", "v")
                )
                names.append(f"{prefix}.layer.1.layer_norm.weight")
                ffn_layer = 2
            else:
                ffn_layer = 1
            names.extend(
                f"{prefix}.layer.{ffn_layer}.DenseReluDense.{projection}.weight"
                for projection in ("wi_0", "wi_1", "wo")
            )
            names.append(f"{prefix}.layer.{ffn_layer}.layer_norm.weight")
    names.extend(
        [
            "decoder.final_layer_norm.weight",
            "encoder.final_layer_norm.weight",
            "lm_head.weight",
            "shared.weight",
        ]
    )
    assert len(names) == 282
    return names


def _write_truth(path: Path, tree_id: str) -> None:
    rows = [
        {"tree_id": tree_id, "node_id": "x", "path": ["root", "x"]},
        {"tree_id": tree_id, "node_id": "a", "path": ["root", "x", "a"]},
        {"tree_id": tree_id, "node_id": "b", "path": ["root", "x", "b"]},
        {"tree_id": tree_id, "node_id": "y", "path": ["root", "y"]},
        {"tree_id": tree_id, "node_id": "c", "path": ["root", "y", "c"]},
        {"tree_id": tree_id, "node_id": "d", "path": ["root", "y", "d"]},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _pinned(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha256(path)}


def _native_summary(
    tmp_path: Path,
    *,
    truth_path: Path,
    distance_layers: Path,
    metrics: list[str],
) -> dict[str, Any]:
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("ledger\n")
    tree = tmp_path / "tree_cosine.newick"
    score = tmp_path / "score_cosine.json"
    audit = tmp_path / "tree_cosine.audit.json"
    matrix_path = tmp_path / "distance_matrix_cosine.npy"
    with np.load(distance_layers, allow_pickle=False) as archive:
        np.save(matrix_path, np.mean(np.asarray(archive["cosine"]), axis=0))
    tree.write_text("((a,b),(c,d));\n")
    _write_json(score, {"valid": True})
    _write_json(audit, {"analysis_engine": "direct"})
    return {
        "schema": DIRECT_TREE_ANALYSIS_SCHEMA,
        "valid": True,
        "producer": "weighttraits",
        "analysis_engine": "direct",
        "distance_engine": "direct_streaming_sufficient_stats",
        "tree_builder": "biopython_neighbor_joining",
        "rf_engine": "dendropy_treecompare",
        "artifact": "model",
        "representation": "full_weight",
        "ledger": str(ledger),
        "ledger_sha256": _sha256(ledger),
        "truth_manifest": str(truth_path),
        "truth_manifest_sha256": _sha256(truth_path),
        "distance_layers": str(distance_layers),
        "n_models": 4,
        "n_layers": 282,
        "model_ids": ["a", "b", "c", "d"],
        "metrics": metrics,
        "results": [
            {
                "metric": "cosine",
                "tree": str(tree),
                "score": str(score),
                "tree_audit": str(audit),
            }
        ],
        "output_sha256": {
            "distance_layers": _sha256(distance_layers),
            "layers": _sha256(tmp_path / "layers.json"),
            "models": _sha256(tmp_path / "models.json"),
            "distance_matrix_cosine": _sha256(matrix_path),
            "tree_cosine": _sha256(tree),
            "score_cosine": _sha256(score),
            "tree_audit_cosine": _sha256(audit),
        },
    }


def _write_completion_receipt(path: Path, *, cohort_id: str) -> None:
    rows = []
    for index, tree_id in enumerate(ALL_TREE_IDS):
        n_runs = 53 if index == len(ALL_TREE_IDS) - 1 else 12
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
    _write_json(
        path,
        {
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
        },
    )


def _write_structural_rollup(path: Path) -> None:
    rows = [
        {"tree_id": tree_id, "metric": metric}
        for tree_id in ALL_TREE_IDS
        for metric in ("correlation", "cosine", "l2")
    ]
    _write_json(
        path,
        {
            "valid": True,
            "artifact": "model",
            "metrics": ["correlation", "cosine", "l2"],
            "missing_tree_ids": [],
            "n_tree_summaries": 50,
            "n_rows": 150,
            "tree_ids": list(ALL_TREE_IDS),
            "rows": rows,
        },
    )


def _make_case(tmp_path: Path) -> Path:
    layer_names = _flan_layer_names()
    layers_path = tmp_path / "layers.json"
    models_path = tmp_path / "models.json"
    cube_path = tmp_path / "direct_distance_layers.npz"
    _write_json(layers_path, layer_names)
    _write_json(models_path, ["a", "b", "c", "d"])
    matrix = np.asarray(
        [
            [0.0, 1.0, 4.0, 4.0],
            [1.0, 0.0, 4.0, 4.0],
            [4.0, 4.0, 0.0, 1.0],
            [4.0, 4.0, 1.0, 0.0],
        ]
    )
    np.savez_compressed(cube_path, cosine=np.repeat(matrix[None, :, :], 282, axis=0))

    truth_specs: dict[str, dict[str, str]] = {}
    inventory_trees: dict[str, dict[str, Any]] = {}
    for tree_id in TOPOLOGY_TREE_IDS:
        truth_path = tmp_path / f"{tree_id}.manifest.jsonl"
        _write_truth(truth_path, tree_id)
        truth_specs[tree_id] = _pinned(truth_path)
        summary_path = tmp_path / f"{tree_id}.summary.json"
        _write_json(
            summary_path,
            _native_summary(
                tmp_path,
                truth_path=truth_path,
                distance_layers=cube_path,
                metrics=["cosine"],
            ),
        )
        inventory_trees[tree_id] = {
            "summary": _pinned(summary_path),
            "layers": _pinned(layers_path),
            "models": _pinned(models_path),
            "distance_layers": _pinned(cube_path),
            "truth_manifest": _pinned(truth_path),
        }

    inventory_path = tmp_path / "inventory.json"
    rollup_path = tmp_path / "rollup.json"
    _write_structural_rollup(rollup_path)
    completion_path = tmp_path / "completion.json"
    _write_completion_receipt(completion_path, cohort_id="flan_full_finetune")
    _write_json(
        inventory_path,
        {
            "schema": LAYER_SUBSET_INVENTORY_SCHEMA,
            "valid": True,
            "cohort_id": "flan_full_finetune",
            "artifact": "model",
            "representation": "full_weight",
            "metric": "cosine",
            "source_rollup": _pinned(rollup_path),
            "completion_receipt": _pinned(completion_path),
            "trees": inventory_trees,
        },
    )
    truth_digest = truth_hashes_sha256(
        {
            tree_id: {"sha256": spec["sha256"]}
            for tree_id, spec in truth_specs.items()
        }
    )
    table2_path = tmp_path / "table2.json"
    _write_json(
        table2_path,
        {
            "schema": DIRECT_WEIGHT_TABLE_SCHEMA,
            "producer": "weighttraits",
            "truth_hashes_sha256": truth_digest,
            "topology_tree_ids": list(TOPOLOGY_TREE_IDS),
            "rows": [
                {
                    "cohort_id": "flan_full_finetune",
                    "metric": "cosine",
                    "artifact": "model",
                    "representation": "full_weight",
                    "n_recovery": 46,
                    "clade_recovery": 1.0,
                    "clade_recovery_se": 0.0,
                    "paer": 1.0,
                    "paer_se": 0.0,
                    "rf": 0.0,
                    "rf_se": 0.0,
                    "false_negative": 0.0,
                    "false_negative_se": 0.0,
                }
            ],
        },
    )
    config = {
        "version": 1,
        "metric": "cosine",
        "cohort_id": "flan_full_finetune",
        "table2_full_ft_cohort_id": "flan_full_finetune",
        "truth_manifests": truth_specs,
        "analysis_inventory": _pinned(inventory_path),
        "table2_receipt": _pinned(table2_path),
    }
    config_path = tmp_path / "table4.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path


def _make_inventory_source_case(tmp_path: Path) -> tuple[Path, Path]:
    layer_names = _flan_layer_names()
    layers_path = tmp_path / "layers.json"
    models_path = tmp_path / "models.json"
    cube_path = tmp_path / "direct_distance_layers.npz"
    _write_json(layers_path, layer_names)
    _write_json(models_path, ["a", "b", "c", "d"])
    matrix = np.asarray(
        [
            [0.0, 1.0, 4.0, 4.0],
            [1.0, 0.0, 4.0, 4.0],
            [4.0, 4.0, 0.0, 1.0],
            [4.0, 4.0, 1.0, 0.0],
        ]
    )
    np.savez_compressed(cube_path, cosine=np.repeat(matrix[None, :, :], 282, axis=0))
    rows = []
    for tree_id in ALL_TREE_IDS:
        truth_path = tmp_path / f"{tree_id}.manifest.jsonl"
        _write_truth(truth_path, tree_id)
        summary_path = tmp_path / f"{tree_id}.summary.json"
        _write_json(
            summary_path,
            _native_summary(
                tmp_path,
                truth_path=truth_path,
                distance_layers=cube_path,
                metrics=["correlation", "cosine", "l2"],
            ),
        )
        for metric in ("correlation", "cosine", "l2"):
            rows.append(
                {
                    "analysis_engine": "direct",
                    "artifact": "model",
                    "representation": "full_weight",
                    "tree_id": tree_id,
                    "metric": metric,
                    "n_layers": 282,
                    "n_models": 4,
                    "summary": str(summary_path),
                    "truth_manifest": str(truth_path),
                }
            )
    rollup_path = tmp_path / "producer-rollup.json"
    _write_json(
        rollup_path,
        {
            "valid": True,
            "artifact": "model",
            "metrics": ["correlation", "cosine", "l2"],
            "missing_tree_ids": [],
            "n_tree_summaries": 50,
            "n_rows": 150,
            "tree_ids": list(ALL_TREE_IDS),
            "rows": rows,
        },
    )
    receipt_path = tmp_path / "producer-completion.json"
    _write_completion_receipt(receipt_path, cohort_id="flan_full_finetune")
    return rollup_path, receipt_path


def test_resolve_layer_subsets_has_exact_documented_counts() -> None:
    resolved = resolve_layer_subsets(_flan_layer_names())
    assert {key: len(value) for key, value in resolved.items()} == EXPECTED_SUBSET_COUNTS


def test_resolve_layer_subsets_rejects_near_match() -> None:
    names = _flan_layer_names()
    names[names.index("encoder.block.0.layer.0.SelfAttention.k.weight")] = (
        "encoder.block.0.layer.0.SelfAttention.key.weight"
    )
    with pytest.raises(ValueError, match="high_signal|enc_sak|sak"):
        resolve_layer_subsets(names)


def test_build_layer_subset_inventory_pins_exact_46_inputs(tmp_path: Path) -> None:
    rollup_path, receipt_path = _make_inventory_source_case(tmp_path)
    payload = build_layer_subset_inventory(
        rollup_path,
        receipt_path,
        cohort_id="flan_full_finetune",
        base_dir=tmp_path,
    )
    assert payload["schema"] == LAYER_SUBSET_INVENTORY_SCHEMA
    assert payload["valid"] is True
    assert payload["topology_tree_ids"] == list(TOPOLOGY_TREE_IDS)
    assert set(payload["trees"]) == set(TOPOLOGY_TREE_IDS)
    assert all(
        set(spec) == {
            "summary",
            "layers",
            "models",
            "distance_layers",
            "truth_manifest",
        }
        for spec in payload["trees"].values()
    )
    output = tmp_path / "published-inventory.json"
    write_layer_subset_inventory_json(payload, output)
    assert json.loads(output.read_text()) == payload


def test_build_layer_subset_inventory_rejects_incomplete_training(
    tmp_path: Path,
) -> None:
    rollup_path, receipt_path = _make_inventory_source_case(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    receipt["n_ready"] = 49
    _write_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="n_ready=50"):
        build_layer_subset_inventory(
            rollup_path,
            receipt_path,
            cohort_id="flan_full_finetune",
            base_dir=tmp_path,
        )


def test_build_layer_subset_table_scores_exact_46_and_matches_table2(tmp_path: Path) -> None:
    payload = build_layer_subset_table(_make_case(tmp_path))

    assert payload["schema"] == LAYER_SUBSET_TABLE_SCHEMA
    assert payload["valid"] is True
    assert payload["topology_tree_ids"] == list(TOPOLOGY_TREE_IDS)
    assert payload["n_topology_trees"] == 46
    assert payload["n_observations"] == 46 * 6
    assert payload["table2_full_row_match"] is True
    assert {row["subset_id"]: row["n_tensors"] for row in payload["rows"]} == (
        EXPECTED_SUBSET_COUNTS
    )
    assert all(row["clade_recovery"] == 1.0 for row in payload["rows"])
    assert all(row["paer"] == 1.0 for row in payload["rows"])


def test_build_layer_subset_table_rejects_same_size_wrong_tree_set(tmp_path: Path) -> None:
    config_path = _make_case(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    inventory_path = Path(config["analysis_inventory"]["path"])
    inventory = json.loads(inventory_path.read_text())
    inventory["trees"]["confirm_paper_tree_015"] = inventory["trees"].pop(
        "confirm_paper_tree_014"
    )
    _write_json(inventory_path, inventory)
    config["analysis_inventory"]["sha256"] = _sha256(inventory_path)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    with pytest.raises(ValueError, match="exactly the canonical 46"):
        build_layer_subset_table(config_path)


def test_build_layer_subset_table_rejects_table2_metric_drift(tmp_path: Path) -> None:
    config_path = _make_case(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    table2_path = Path(config["table2_receipt"]["path"])
    table2 = json.loads(table2_path.read_text())
    table2["rows"][0]["clade_recovery"] = 0.99
    _write_json(table2_path, table2)
    config["table2_receipt"]["sha256"] = _sha256(table2_path)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    with pytest.raises(ValueError, match="does not match corrected Table 2"):
        build_layer_subset_table(config_path)


def test_build_layer_subset_table_rejects_summary_truth_at_identical_copy_path(
    tmp_path: Path,
) -> None:
    config_path = _make_case(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    inventory_path = Path(config["analysis_inventory"]["path"])
    inventory = json.loads(inventory_path.read_text())
    tree_id = TOPOLOGY_TREE_IDS[0]
    summary_path = Path(inventory["trees"][tree_id]["summary"]["path"])
    summary = json.loads(summary_path.read_text())
    original = Path(config["truth_manifests"][tree_id]["path"])
    alternate = tmp_path / "identical-copy.manifest.jsonl"
    alternate.write_bytes(original.read_bytes())
    summary["truth_manifest"] = str(alternate)
    _write_json(summary_path, summary)
    inventory["trees"][tree_id]["summary"]["sha256"] = _sha256(summary_path)
    _write_json(inventory_path, inventory)
    config["analysis_inventory"]["sha256"] = _sha256(inventory_path)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    with pytest.raises(ValueError, match="truth path mismatch"):
        build_layer_subset_table(config_path)
