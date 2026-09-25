from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from weighttraits.behavior.phylolm import (
    LOCKED_PHYLOLM_PROTOCOL,
    PHYLOLM_GENERATION_SCHEMA,
    PHYLOLM_GENOME_SCHEMA,
    PHYLOLM_POPULATION_SCHEMA,
    PHYLOLM_TREE_SCHEMA,
    PhyloLMProtocol,
    analyze_phylolm_tree,
    build_phylolm_runset_receipt,
    build_population_receipt,
    build_sampled_genome,
    compute_population,
    nei_distance_matrix,
    write_genome_receipt,
    write_population_receipt,
    _validate_tree_receipt,
)
from weighttraits.paper.analysis_contracts import (
    ALL_TREE_IDS,
    TOPOLOGY_TREE_IDS,
    canonical_sha256,
)
from weighttraits.paper.phylolm_table import PHYLOLM_RECEIPT_SCHEMA


SUITE = "llama32_1b_full_finetune_legacy_causal_2000"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _make_genome(tmp_path: Path) -> Path:
    pool = tmp_path / "genes.json"
    pool.write_text(json.dumps([f"gene-{index}" for index in range(256)]) + "\n")
    payload = build_sampled_genome(pool)
    assert payload["schema"] == PHYLOLM_GENOME_SCHEMA
    genome = tmp_path / "genome.json"
    write_genome_receipt(payload, genome)
    return genome


def _population(allele: str) -> list[dict[str, float]]:
    return compute_population(
        [[allele for _ in range(32)] for _ in range(128)]
    )


def _generation_receipt(
    tmp_path: Path,
    *,
    tree_id: str,
    model_id: str,
    truth: Path,
    checkpoint: Path,
    completion: Path,
    stage: Path,
) -> dict[str, Any]:
    summary = tmp_path / "training-summary.json"
    run_list = tmp_path / "training-runs.jsonl"
    ledger = tmp_path / "training-ledger.jsonl"
    for path in (summary, run_list, ledger):
        if not path.exists():
            path.write_text(f"{path.name}\n")
    checkpoint_digest = _sha(checkpoint)
    return {
        "schema": PHYLOLM_GENERATION_SCHEMA,
        "valid": True,
        "suite_id": SUITE,
        "tree_id": tree_id,
        "model_id": model_id,
        "protocol": LOCKED_PHYLOLM_PROTOCOL.to_dict(),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_digest,
        "training_summary": str(summary.resolve()),
        "training_summary_sha256": _sha(summary),
        "run_list": str(run_list.resolve()),
        "run_list_sha256": _sha(run_list),
        "ledger": str(ledger.resolve()),
        "ledger_sha256": _sha(ledger),
        "truth_manifest": str(truth.resolve()),
        "truth_manifest_sha256": _sha(truth),
        "completion_receipt": str(completion.resolve()),
        "completion_receipt_sha256": _sha(completion),
        "stage_manifest": str(stage.resolve()),
        "stage_manifest_sha256": _sha(stage),
        "loader": {
            "backend": "huggingface_text_generation_pipeline",
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_artifact": "model",
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


def _write_completion(path: Path) -> None:
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
    path.write_text(
        json.dumps(
            {
                "cohort_id": SUITE,
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
        )
        + "\n"
    )


def test_locked_genome_and_population_contract(tmp_path: Path) -> None:
    genome = _make_genome(tmp_path)
    first = json.loads(genome.read_text())
    second = build_sampled_genome(Path(first["source_gene_pool"]))
    assert second["selected_indices"] == first["selected_indices"]
    assert second["genes_sha256"] == first["genes_sha256"]
    with pytest.raises(ValueError, match="locked"):
        build_sampled_genome(
            Path(first["source_gene_pool"]),
            protocol=PhyloLMProtocol(sampling_seed=0),
        )
    with pytest.raises(ValueError, match="locked"):
        build_sampled_genome(
            Path(first["source_gene_pool"]),
            protocol=PhyloLMProtocol(batch_size=32),
        )
    with pytest.raises(ValueError, match="exactly 32"):
        compute_population([["AAAA"] * 31 for _ in range(128)])


def test_genome_receipt_replays_indices_and_source_genes(tmp_path: Path) -> None:
    genome = _make_genome(tmp_path)
    payload = json.loads(genome.read_text())
    payload["selected_indices"][0] = -1
    payload["selected_indices_sha256"] = canonical_sha256(payload["selected_indices"])
    payload["genes"][0] = "forged-gene"
    payload["genes_sha256"] = canonical_sha256(payload["genes"])
    with pytest.raises(ValueError, match="outside|replay"):
        write_genome_receipt(payload, tmp_path / "forged-genome.json")


def test_tree_analysis_uses_pinned_populations_and_scores_truth(tmp_path: Path) -> None:
    genome = _make_genome(tmp_path)
    truth = tmp_path / "truth.jsonl"
    tree_id = TOPOLOGY_TREE_IDS[0]
    _write_truth(truth, tree_id)
    completion = tmp_path / "completion.json"
    stage = tmp_path / "stage.sha256"
    _write_completion(completion)
    stage.write_text("stage\n")
    specs = {}
    for model_id, allele in (("a", "AAAA"), ("b", "AAAA"), ("c", "CCCC"), ("d", "CCCC")):
        checkpoint = tmp_path / f"{model_id}.checkpoint"
        checkpoint.write_text(model_id)
        payload = build_population_receipt(
            suite_id=SUITE,
            tree_id=tree_id,
            model_id=model_id,
            population=_population(allele),
            genome_receipt_path=genome,
            checkpoint_path=checkpoint,
            checkpoint_sha256=_sha(checkpoint),
            generation_receipt=_generation_receipt(
                tmp_path,
                tree_id=tree_id,
                model_id=model_id,
                truth=truth,
                checkpoint=checkpoint,
                completion=completion,
                stage=stage,
            ),
        )
        assert payload["schema"] == PHYLOLM_POPULATION_SCHEMA
        population_path = tmp_path / f"{model_id}.population.json"
        write_population_receipt(payload, population_path)
        specs[model_id] = {"path": str(population_path), "sha256": _sha(population_path)}

    receipt = analyze_phylolm_tree(
        specs,
        suite_id=SUITE,
        tree_id=tree_id,
        truth_manifest_path=truth,
        output_dir=tmp_path / "analysis",
    )
    assert receipt["schema"] == PHYLOLM_TREE_SCHEMA
    assert receipt["valid"] is True
    assert receipt["n_truth_splits"] > 0
    assert receipt["false_negative"] == 0
    assert receipt["polytomy_aware_exact_recovery"] is True
    assert Path(receipt["outputs"]["distance"]["path"]).is_file()

    similarity_path = Path(receipt["outputs"]["similarity"]["path"])
    distance_path = Path(receipt["outputs"]["distance"]["path"])
    forged_similarity = np.full((4, 4), 0.5, dtype=np.float64)
    np.fill_diagonal(forged_similarity, 1.0)
    forged_distance = -np.log(np.maximum(forged_similarity, 1e-3))
    np.fill_diagonal(forged_distance, 0.0)
    np.save(similarity_path, forged_similarity)
    np.save(distance_path, forged_distance)
    receipt["outputs"]["similarity"]["sha256"] = _sha(similarity_path)
    receipt["outputs"]["distance"]["sha256"] = _sha(distance_path)
    with pytest.raises(ValueError, match="population replay"):
        _validate_tree_receipt(receipt, suite_id=SUITE, tree_id=tree_id)


def test_nei_distance_fails_wrong_gene_count() -> None:
    with pytest.raises(ValueError, match="128"):
        nei_distance_matrix({name: [{"AAAA": 1.0}] for name in ("a", "b", "c", "d")})


def test_runset_receipt_requires_exact_common_46(tmp_path: Path) -> None:
    completion = tmp_path / "completion.json"
    stage = tmp_path / "stage.sha256"
    _write_completion(completion)
    stage.write_text("stage\n")
    genome = _make_genome(tmp_path)
    checkpoint = tmp_path / "shared.checkpoint"
    checkpoint.write_text("checkpoint")
    tree_specs = {}
    for tree_id in TOPOLOGY_TREE_IDS:
        tree_dir = tmp_path / tree_id
        tree_dir.mkdir()
        truth = tree_dir / "truth.jsonl"
        _write_truth(truth, tree_id)
        populations = {}
        for model_id, allele in (
            ("a", "AAAA"),
            ("b", "AAAA"),
            ("c", "CCCC"),
            ("d", "CCCC"),
        ):
            population = build_population_receipt(
                suite_id=SUITE,
                tree_id=tree_id,
                model_id=model_id,
                population=_population(allele),
                genome_receipt_path=genome,
                checkpoint_path=checkpoint,
                checkpoint_sha256=_sha(checkpoint),
                generation_receipt=_generation_receipt(
                    tree_dir,
                    tree_id=tree_id,
                    model_id=model_id,
                    truth=truth,
                    checkpoint=checkpoint,
                    completion=completion,
                    stage=stage,
                ),
            )
            population_path = tree_dir / f"{model_id}.population.json"
            write_population_receipt(population, population_path)
            populations[model_id] = {
                "path": str(population_path),
                "sha256": _sha(population_path),
            }
        analysis_dir = tree_dir / "analysis"
        analyze_phylolm_tree(
            populations,
            suite_id=SUITE,
            tree_id=tree_id,
            truth_manifest_path=truth,
            output_dir=analysis_dir,
        )
        tree_receipt = analysis_dir / "receipt.json"
        tree_specs[tree_id] = {
            "path": str(tree_receipt),
            "sha256": _sha(tree_receipt),
        }
    payload = build_phylolm_runset_receipt(
        tree_specs,
        suite_id=SUITE,
        completion_receipt={"path": str(completion), "sha256": _sha(completion)},
        stage_manifest={"path": str(stage), "sha256": _sha(stage)},
    )
    assert payload["schema"] == PHYLOLM_RECEIPT_SCHEMA
    assert payload["valid"] is True
    assert payload["n_trees"] == 46
    assert payload["group"] == f"runs_weighttraits_{SUITE}_fresh_phylolm"
    missing = dict(tree_specs)
    missing.pop(TOPOLOGY_TREE_IDS[-1])
    with pytest.raises(ValueError, match="exact canonical"):
        build_phylolm_runset_receipt(
            missing,
            suite_id=SUITE,
            completion_receipt={"path": str(completion), "sha256": _sha(completion)},
            stage_manifest={"path": str(stage), "sha256": _sha(stage)},
        )
