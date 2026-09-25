from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from weighttraits.behavior.phylolm import (
    LOCKED_PHYLOLM_PROTOCOL,
    PHYLOLM_GENERATION_SCHEMA,
    analyze_phylolm_tree,
    build_phylolm_runset_receipt,
    build_population_receipt,
    build_sampled_genome,
    compute_population,
    write_genome_receipt,
    write_population_receipt,
)
from weighttraits.paper.analysis_contracts import (
    ALL_TREE_IDS,
    TOPOLOGY_EXCLUDED_TREE_IDS,
    TOPOLOGY_TREE_IDS,
    truth_hashes_sha256,
)
from weighttraits.paper.phylolm_table import (
    DIRECT_TREE_ANALYSIS_SCHEMA,
    PHYLOLM_PROTOCOL_CONTRACT,
    PHYLOLM_RECEIPT_SCHEMA,
    PHYLOLM_TABLE_SCHEMA,
    SUITE_CONTRACTS,
    build_phylolm_table,
    resolve_legacy_causal_suite,
)
from weighttraits.phylo.reconstruct import neighbor_joining_newick
from weighttraits.phylo.recovery import score_split_recovery
from weighttraits.phylo.splits import splits_from_manifest_path, splits_from_newick_text


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _pinned(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha256(path)}


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


def _recovery_row(index: int, *, phylolm: bool = False) -> dict[str, Any]:
    false_negative = 1 if (index + (1 if phylolm else 0)) % 3 == 0 else 0
    false_positive = index % 2
    return {
        "n_truth_splits": 2,
        "false_negative": false_negative,
        "false_positive": false_positive,
        "rf": false_negative + false_positive,
        "clade_recovery": (2 - false_negative) / 2,
        "polytomy_aware_exact_recovery": false_negative == 0,
    }


def _population(allele: str) -> list[dict[str, float]]:
    return compute_population([[allele] * 32 for _ in range(128)])


def _generation_receipt(
    *,
    suite_id: str,
    tree_id: str,
    model_id: str,
    checkpoint: Path,
    truth: Path,
    completion: Path,
    stage: Path,
    summary: Path,
    run_list: Path,
    ledger: Path,
    artifact: str,
) -> dict[str, Any]:
    return {
        "schema": PHYLOLM_GENERATION_SCHEMA,
        "valid": True,
        "suite_id": suite_id,
        "tree_id": tree_id,
        "model_id": model_id,
        "protocol": LOCKED_PHYLOLM_PROTOCOL.to_dict(),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "training_summary": str(summary.resolve()),
        "training_summary_sha256": _sha256(summary),
        "run_list": str(run_list.resolve()),
        "run_list_sha256": _sha256(run_list),
        "ledger": str(ledger.resolve()),
        "ledger_sha256": _sha256(ledger),
        "truth_manifest": str(truth.resolve()),
        "truth_manifest_sha256": _sha256(truth),
        "completion_receipt": str(completion.resolve()),
        "completion_receipt_sha256": _sha256(completion),
        "stage_manifest": str(stage.resolve()),
        "stage_manifest_sha256": _sha256(stage),
        "loader": {
            "backend": "huggingface_text_generation_pipeline",
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_artifact": artifact,
            "tokenizer_model_id": "meta-llama/Llama-3.2-1B",
            "tokenizer_revision": "4e20de36255befb501235df90d39eebbac5bd237",
            "local_files_only": True,
            "torch_dtype": "auto",
            "torch_version": "test",
            "transformers_version": "test",
        },
        "batch_size": 64,
        "torch_dtype": "auto",
        "elapsed_seconds": 1.0,
    }


def _native_weight_row(
    root: Path,
    *,
    tree_id: str,
    artifact: str,
    truth: Path,
) -> dict[str, Any]:
    analysis = root / tree_id / "weight"
    analysis.mkdir(parents=True)
    model_ids = ["a", "b", "c", "d"]
    matrix = np.asarray(
        [
            [0.0, 1.0, 4.0, 4.0],
            [1.0, 0.0, 4.0, 4.0],
            [4.0, 4.0, 0.0, 1.0],
            [4.0, 4.0, 1.0, 0.0],
        ]
    )
    cube = analysis / "direct_distance_layers.npz"
    np.savez_compressed(cube, cosine=matrix[None, :, :])
    _write_json(analysis / "layers.json", ["layer.weight"])
    _write_json(analysis / "models.json", model_ids)
    np.save(analysis / "distance_matrix_cosine.npy", matrix)
    tree_path = analysis / "tree_cosine.newick"
    tree_path.write_text(neighbor_joining_newick(model_ids, matrix) + "\n")
    truth_splits, truth_leaves = splits_from_manifest_path(str(truth))
    estimate_splits, estimate_leaves = splits_from_newick_text(tree_path.read_text())
    score = score_split_recovery(
        truth_splits, estimate_splits, truth_leaves, estimate_leaves
    )
    score_path = analysis / "score_cosine.json"
    _write_json(score_path, score)
    audit_path = analysis / "tree_cosine.audit.json"
    _write_json(audit_path, {"analysis_engine": "direct"})
    ledger = analysis / "ledger.jsonl"
    ledger.write_text("ledger\n")
    result = {
        "metric": "cosine",
        "tree": str(tree_path),
        "tree_audit": str(audit_path),
        "score": str(score_path),
        **score,
    }
    summary_path = analysis / "summary.json"
    _write_json(
        summary_path,
        {
            "schema": DIRECT_TREE_ANALYSIS_SCHEMA,
            "valid": True,
            "producer": "weighttraits",
            "analysis_engine": "direct",
            "distance_engine": "direct_streaming_sufficient_stats",
            "tree_builder": "biopython_neighbor_joining",
            "rf_engine": "dendropy_treecompare",
            "artifact": artifact,
            "representation": "full_weight",
            "ledger": str(ledger),
            "ledger_sha256": _sha256(ledger),
            "truth_manifest": str(truth),
            "truth_manifest_sha256": _sha256(truth),
            "distance_layers": str(cube),
            "n_models": len(model_ids),
            "n_layers": 1,
            "model_ids": model_ids,
            "metrics": ["cosine"],
            "results": [result],
            "output_sha256": {
                "distance_layers": _sha256(cube),
                "layers": _sha256(analysis / "layers.json"),
                "models": _sha256(analysis / "models.json"),
                "distance_matrix_cosine": _sha256(
                    analysis / "distance_matrix_cosine.npy"
                ),
                "tree_cosine": _sha256(tree_path),
                "score_cosine": _sha256(score_path),
                "tree_audit_cosine": _sha256(audit_path),
            },
        },
    )
    return {
        "tree_id": tree_id,
        "metric": "cosine",
        "artifact": artifact,
        "representation": "full_weight",
        "analysis_engine": "direct",
        "tree_builder": "biopython_neighbor_joining",
        "rf_engine": "dendropy_treecompare",
        "summary": str(summary_path),
        "score": str(score_path),
        "ledger": str(ledger),
        "truth_manifest": str(truth),
        **score,
    }


def _make_case(tmp_path: Path) -> Path:
    truth_specs: dict[str, dict[str, str]] = {}
    for tree_id in TOPOLOGY_TREE_IDS:
        path = tmp_path / f"{tree_id}.manifest.jsonl"
        _write_truth(path, tree_id)
        truth_specs[tree_id] = _pinned(path)
    truth_map = {tree_id: spec["sha256"] for tree_id, spec in truth_specs.items()}
    truth_digest = truth_hashes_sha256(
        {tree_id: {"sha256": digest} for tree_id, digest in truth_map.items()}
    )
    gene_pool = tmp_path / "genes.json"
    _write_json(gene_pool, [f"gene-{index}" for index in range(256)])
    genome_path = tmp_path / "genome.json"
    write_genome_receipt(build_sampled_genome(gene_pool), genome_path)

    suites = []
    for suite_id, contract in SUITE_CONTRACTS.items():
        suite_root = tmp_path / suite_id
        suite_root.mkdir()
        cohort_path = tmp_path / f"{suite_id}.cohort.json"
        completion_path = tmp_path / f"{suite_id}.completion.json"
        cohort_trees = []
        for tree_id in ALL_TREE_IDS:
            cohort_trees.append(
                {
                    "tree_id": tree_id,
                    "valid": True,
                    "manifest_sha256": truth_map.get(tree_id, "0" * 64),
                }
            )
        _write_json(
            cohort_path,
            {
                "valid": True,
                "n_trees": 50,
                "n_runs": 641,
                "n_errors": 0,
                "n_warnings": 0,
                "config": f"examples/{suite_id}.yaml",
                "trees": cohort_trees,
            },
        )
        _write_completion_receipt(completion_path, cohort_id=suite_id)

        weight_path = tmp_path / f"{suite_id}.weight.json"
        weight_rows = []
        for tree_id in ALL_TREE_IDS:
            excluded = tree_id in TOPOLOGY_EXCLUDED_TREE_IDS
            if excluded:
                row = {
                    "tree_id": tree_id,
                    "metric": "cosine",
                    "artifact": contract["weight_artifact"],
                    "representation": "full_weight",
                    "n_truth_splits": 0,
                    "false_negative": 0,
                    "false_positive": 0,
                    "rf": 0,
                    "clade_recovery": 1.0,
                    "polytomy_aware_exact_recovery": True,
                }
            else:
                row = _native_weight_row(
                    suite_root,
                    tree_id=tree_id,
                    artifact=contract["weight_artifact"],
                    truth=Path(truth_specs[tree_id]["path"]),
                )
            weight_rows.append(row)
        _write_json(
            weight_path,
            {
                "valid": True,
                "artifact": contract["weight_artifact"],
                "rows": weight_rows,
            },
        )

        phylolm_path = tmp_path / f"{suite_id}.phylolm.json"
        stage_path = tmp_path / f"{suite_id}.stage.sha256"
        stage_path.write_text("stage\n")
        summary = suite_root / "training-summary.json"
        run_list = suite_root / "training-runs.jsonl"
        ledger = suite_root / "training-ledger.jsonl"
        for source in (summary, run_list, ledger):
            source.write_text(f"{source.name}\n")
        checkpoints = {}
        for model_id in ("a", "b", "c", "d"):
            checkpoint = suite_root / f"{model_id}.checkpoint"
            checkpoint.write_text(f"{model_id}\n")
            checkpoints[model_id] = checkpoint
        tree_receipts = {}
        for tree_id in TOPOLOGY_TREE_IDS:
            tree_root = suite_root / tree_id
            tree_root.mkdir(exist_ok=True)
            populations = {}
            for model_id, allele in (
                ("a", "AAAA"),
                ("b", "AAAA"),
                ("c", "CCCC"),
                ("d", "CCCC"),
            ):
                checkpoint = checkpoints[model_id]
                population = build_population_receipt(
                    suite_id=suite_id,
                    tree_id=tree_id,
                    model_id=model_id,
                    population=_population(allele),
                    genome_receipt_path=genome_path,
                    checkpoint_path=checkpoint,
                    checkpoint_sha256=_sha256(checkpoint),
                    generation_receipt=_generation_receipt(
                        suite_id=suite_id,
                        tree_id=tree_id,
                        model_id=model_id,
                        checkpoint=checkpoint,
                        truth=Path(truth_specs[tree_id]["path"]),
                        completion=completion_path,
                        stage=stage_path,
                        summary=summary,
                        run_list=run_list,
                        ledger=ledger,
                        artifact=contract["weight_artifact"],
                    ),
                )
                population_path = tree_root / f"{model_id}.population.json"
                write_population_receipt(population, population_path)
                populations[model_id] = _pinned(population_path)
            analysis_dir = tree_root / "analysis"
            analyze_phylolm_tree(
                populations,
                suite_id=suite_id,
                tree_id=tree_id,
                truth_manifest_path=truth_specs[tree_id]["path"],
                output_dir=analysis_dir,
            )
            tree_receipts[tree_id] = _pinned(analysis_dir / "receipt.json")
        runset = build_phylolm_runset_receipt(
            tree_receipts,
            suite_id=suite_id,
            completion_receipt=_pinned(completion_path),
            stage_manifest=_pinned(stage_path),
        )
        _write_json(phylolm_path, runset)
        suites.append(
            {
                "suite_id": suite_id,
                "cohort_contract": _pinned(cohort_path),
                "completion_receipt": _pinned(completion_path),
                "weight_rollup": _pinned(weight_path),
                "phylolm_receipt": _pinned(phylolm_path),
            }
        )

    config_path = tmp_path / "table10.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "metric": "cosine",
                "truth_manifests": truth_specs,
                "suites": suites,
            },
            sort_keys=False,
        )
    )
    return config_path


def _mutate_phylolm_receipt(
    config_path: Path,
    suite_index: int,
    mutator: Any,
) -> None:
    config = yaml.safe_load(config_path.read_text())
    spec = config["suites"][suite_index]["phylolm_receipt"]
    path = Path(spec["path"])
    payload = json.loads(path.read_text())
    mutator(payload)
    _write_json(path, payload)
    spec["sha256"] = _sha256(path)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))


def test_resolve_legacy_causal_suite_is_exact() -> None:
    for suite_id in SUITE_CONTRACTS:
        assert resolve_legacy_causal_suite(suite_id)["suite_id"] == suite_id
    with pytest.raises(ValueError, match="unsupported Table 10 suite"):
        resolve_legacy_causal_suite("llama_r8")


def test_build_phylolm_table_pairs_exact_canonical_46(tmp_path: Path) -> None:
    payload = build_phylolm_table(_make_case(tmp_path))

    assert payload["schema"] == PHYLOLM_TABLE_SCHEMA
    assert payload["valid"] is True
    assert payload["common_tree_ids"] == list(TOPOLOGY_TREE_IDS)
    assert payload["n_common_trees"] == 46
    assert payload["n_rows"] == 3
    assert payload["n_paired_observations"] == 3 * 46
    assert [row["suite_id"] for row in payload["rows"]] == list(SUITE_CONTRACTS)


def test_build_phylolm_table_rejects_same_size_wrong_tree_set(tmp_path: Path) -> None:
    config_path = _make_case(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        row = next(
            row
            for row in payload["rows"]
            if row["tree_id"] == "confirm_paper_tree_014"
        )
        row["tree_id"] = "confirm_paper_tree_015"
        row["truth_sha256"] = "0" * 64

    _mutate_phylolm_receipt(config_path, 0, mutate)
    with pytest.raises(ValueError, match="truth hash mismatch|exact (?:canonical )?common-46"):
        build_phylolm_table(config_path)


def test_build_phylolm_table_rejects_truth_hash_drift(tmp_path: Path) -> None:
    config_path = _make_case(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        payload["rows"][0]["truth_sha256"] = "f" * 64

    _mutate_phylolm_receipt(config_path, 1, mutate)
    with pytest.raises(ValueError, match="truth hash mismatch|provenance differs from tree receipt"):
        build_phylolm_table(config_path)


def test_build_phylolm_table_rejects_metric_inconsistency(tmp_path: Path) -> None:
    config_path = _make_case(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        payload["rows"][0]["clade_recovery"] = 0.123

    _mutate_phylolm_receipt(config_path, 2, mutate)
    with pytest.raises(
        ValueError,
        match="differs from (?:runset row|tree receipt)|inconsistent with truth/FN",
    ):
        build_phylolm_table(config_path)


def test_build_phylolm_table_rejects_incomplete_cohort_receipt(
    tmp_path: Path,
) -> None:
    config_path = _make_case(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    receipt_spec = config["suites"][0]["cohort_contract"]
    receipt_path = Path(receipt_spec["path"])
    receipt = json.loads(receipt_path.read_text())
    receipt["n_runs"] = 640
    _write_json(receipt_path, receipt)
    receipt_spec["sha256"] = _sha256(receipt_path)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    with pytest.raises(ValueError, match="n_runs must be 641"):
        build_phylolm_table(config_path)
