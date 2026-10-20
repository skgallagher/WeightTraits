"""Pinned native PhyloLM population, distance, and controlled-tree receipts.

This module implements the locked black-box comparison used by corrected
Table 10.  It deliberately does not know historical ELLMTrees directory
aliases.  Every population is tied to one corrected suite/tree/leaf,
checkpoint hash, sampled-genome receipt, and the exact upstream protocol.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from weighttraits.manifests.reference import manifest_leaf_ids
from weighttraits.paper.analysis_contracts import (
    TOPOLOGY_TREE_IDS,
    canonical_sha256,
    canonical_tree_id,
    sha256,
    validate_strict_training_completion,
)
from weighttraits.paper.phylolm_table import (
    PHYLOLM_RECEIPT_SCHEMA,
    SUITE_CONTRACTS,
)
from weighttraits.phylo.reconstruct import neighbor_joining_newick
from weighttraits.phylo.recovery import score_split_recovery
from weighttraits.phylo.splits import splits_from_manifest_path, splits_from_newick_text


PHYLOLM_UPSTREAM_COMMIT = "8c70edf062a0adce2a3e6c8c79cd23a645fd0905"
PHYLOLM_GENOME_SCHEMA = "weighttraits.phylolm_genome.v1"
PHYLOLM_GENERATION_SCHEMA = "weighttraits.phylolm_generation.v1"
PHYLOLM_POPULATION_SCHEMA = "weighttraits.phylolm_population.v1"
PHYLOLM_TREE_SCHEMA = "weighttraits.phylolm_controlled_tree.v1"


@dataclass(frozen=True)
class PhyloLMProtocol:
    """Exact protocol locked for the corrected common-46 comparison."""

    upstream_commit: str = PHYLOLM_UPSTREAM_COMMIT
    n_genes: int = 128
    samples_per_gene: int = 32
    new_tokens: int = 4
    allele_characters: int = 4
    temperature: float = 1.0
    gene_seed: int = 0
    sampling_seed: int = 20260803
    batch_size: int = 64
    torch_dtype: str = "auto"
    similarity_floor: float = 1e-3
    prompt_mode: str = "raw_continuation"
    distance: str = "-log(max(nei_similarity,1e-3))"
    tree_builder: str = "weighttraits_neighbor_joining"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


LOCKED_PHYLOLM_PROTOCOL = PhyloLMProtocol()


def load_gene_pool(path: str | Path) -> list[str]:
    source = Path(path)
    raw = json.loads(source.read_text())
    if not isinstance(raw, list) or len(raw) < LOCKED_PHYLOLM_PROTOCOL.n_genes:
        raise ValueError("PhyloLM gene pool must be a JSON list with at least 128 rows")
    if not all(isinstance(item, str) and item for item in raw):
        raise ValueError("PhyloLM gene pool must contain only nonempty strings")
    return list(raw)


def build_sampled_genome(
    gene_pool_path: str | Path,
    *,
    protocol: PhyloLMProtocol = LOCKED_PHYLOLM_PROTOCOL,
) -> dict[str, Any]:
    _require_locked_protocol(protocol)
    source = Path(gene_pool_path).resolve()
    pool = load_gene_pool(source)
    rng = np.random.default_rng(protocol.gene_seed)
    indices = rng.choice(len(pool), size=protocol.n_genes, replace=False)
    selected_indices = [int(index) for index in indices]
    genes = [pool[index] for index in selected_indices]
    return {
        "schema": PHYLOLM_GENOME_SCHEMA,
        "valid": True,
        "protocol": protocol.to_dict(),
        "source_gene_pool": str(source),
        "source_gene_pool_sha256": sha256(source),
        "source_gene_count": len(pool),
        "selected_indices": selected_indices,
        "selected_indices_sha256": canonical_sha256(selected_indices),
        "genes": genes,
        "genes_sha256": canonical_sha256(genes),
    }


def write_genome_receipt(payload: Mapping[str, Any], path: str | Path) -> None:
    _validate_genome_payload(payload)
    _write_json_atomic(Path(path), payload)


def load_genome_receipt(path: str | Path) -> dict[str, Any]:
    """Load and replay-validate the pinned seed-0 sampled genome."""

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"PhyloLM genome receipt is missing: {source}")
    payload = json.loads(source.read_text())
    _validate_genome_payload(payload)
    return dict(payload)


def compute_population(alleles: Sequence[Sequence[str]]) -> list[dict[str, float]]:
    rows = [list(row) for row in alleles]
    protocol = LOCKED_PHYLOLM_PROTOCOL
    if len(rows) != protocol.n_genes:
        raise ValueError(
            f"PhyloLM requires exactly {protocol.n_genes} genes, got {len(rows)}"
        )
    if any(len(row) != protocol.samples_per_gene for row in rows):
        raise ValueError(
            "PhyloLM requires exactly "
            f"{protocol.samples_per_gene} samples for every gene"
        )
    population: list[dict[str, float]] = []
    for row in rows:
        if not all(isinstance(allele, str) for allele in row):
            raise ValueError("PhyloLM alleles must be strings")
        if any(len(allele) > protocol.allele_characters for allele in row):
            raise ValueError("PhyloLM allele exceeds the locked four-character limit")
        counts = Counter(row)
        population.append(
            {
                allele: count / protocol.samples_per_gene
                for allele, count in sorted(counts.items())
            }
        )
    _validate_population(population)
    return population


def build_population_receipt(
    *,
    suite_id: str,
    tree_id: str,
    model_id: str,
    population: Sequence[Mapping[str, float]],
    genome_receipt_path: str | Path,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    generation_receipt: Mapping[str, Any],
    protocol: PhyloLMProtocol = LOCKED_PHYLOLM_PROTOCOL,
) -> dict[str, Any]:
    _require_locked_protocol(protocol)
    _require_suite(suite_id)
    tree = canonical_tree_id(tree_id)
    model = _nonempty(model_id, "model_id")
    genome_path = Path(genome_receipt_path).resolve()
    genome = json.loads(genome_path.read_text())
    _validate_genome_payload(genome)
    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"PhyloLM checkpoint is missing: {checkpoint}")
    if not _is_sha256(checkpoint_sha256):
        raise ValueError("checkpoint_sha256 must be lowercase SHA-256")
    values = [dict(row) for row in population]
    _validate_population(values)
    if not isinstance(generation_receipt, Mapping) or not generation_receipt:
        raise ValueError("PhyloLM generation_receipt must be a nonempty mapping")
    generation = dict(generation_receipt)
    _validate_generation_receipt(
        generation,
        suite_id=suite_id,
        tree_id=tree,
        model_id=model,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
    )
    return {
        "schema": PHYLOLM_POPULATION_SCHEMA,
        "valid": True,
        "suite_id": suite_id,
        "tree_id": tree,
        "model_id": model,
        "protocol": protocol.to_dict(),
        "genome_receipt": str(genome_path),
        "genome_receipt_sha256": sha256(genome_path),
        "genes_sha256": genome["genes_sha256"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "generation_receipt": generation,
        "generation_receipt_sha256": canonical_sha256(generation),
        "population": values,
        "population_sha256": canonical_sha256(values),
    }


def write_population_receipt(payload: Mapping[str, Any], path: str | Path) -> None:
    _validate_population_payload(payload)
    _write_json_atomic(Path(path), payload)


def load_population_receipt(
    path: str | Path,
    *,
    suite_id: str,
    tree_id: str,
    model_id: str,
    genome_receipt_path: str | Path,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    completion_receipt_path: str | Path,
    stage_manifest_path: str | Path,
    checkpoint_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Read and verify one cached population against the exact request inputs."""

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"PhyloLM population receipt is missing: {source}")
    payload = json.loads(source.read_text())
    _validate_population_payload(
        payload,
        suite_id=suite_id,
        tree_id=canonical_tree_id(tree_id),
        model_id=model_id,
    )
    genome = Path(genome_receipt_path).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    completion = Path(completion_receipt_path).resolve()
    stage = Path(stage_manifest_path).resolve()
    if (
        payload.get("genome_receipt") != str(genome)
        or payload.get("genome_receipt_sha256") != sha256(genome)
        or payload.get("checkpoint") != str(checkpoint)
        or payload.get("checkpoint_sha256") != checkpoint_sha256
    ):
        raise ValueError("cached PhyloLM population input provenance drift")
    generation = payload.get("generation_receipt")
    if not isinstance(generation, Mapping):
        raise ValueError("cached PhyloLM population lacks generation receipt")
    expected_pins = {
        "completion_receipt": str(completion),
        "completion_receipt_sha256": sha256(completion),
        "stage_manifest": str(stage),
        "stage_manifest_sha256": sha256(stage),
    }
    for field, expected in expected_pins.items():
        if generation.get(field) != expected:
            raise ValueError(f"cached PhyloLM population {field} drift")
    for field in (
        "training_summary",
        "run_list",
        "ledger",
        "truth_manifest",
    ):
        raw_path = checkpoint_provenance.get(field)
        raw_digest = checkpoint_provenance.get(f"{field}_sha256")
        expected_path = str(Path(str(raw_path)).resolve())
        if (
            generation.get(field) != expected_path
            or generation.get(f"{field}_sha256") != raw_digest
        ):
            raise ValueError(f"cached PhyloLM population {field} provenance drift")
    loader = generation.get("loader")
    if not isinstance(loader, Mapping):
        raise ValueError("cached PhyloLM population lacks loader provenance")
    for field, expected in (
        ("checkpoint_artifact", checkpoint_provenance.get("artifact_name")),
        ("tokenizer_model_id", checkpoint_provenance.get("base_model_id")),
        ("tokenizer_revision", checkpoint_provenance.get("base_model_revision")),
    ):
        if loader.get(field) != expected:
            raise ValueError(f"cached PhyloLM population loader {field} drift")
    return dict(payload)


def analyze_phylolm_tree(
    population_specs: Mapping[str, Mapping[str, str]],
    *,
    suite_id: str,
    tree_id: str,
    truth_manifest_path: str | Path,
    output_dir: str | Path,
    protocol: PhyloLMProtocol = LOCKED_PHYLOLM_PROTOCOL,
) -> dict[str, Any]:
    """Analyze one tree and atomically publish its matrix/tree/score receipt."""

    _require_locked_protocol(protocol)
    _require_suite(suite_id)
    tree = canonical_tree_id(tree_id)
    truth = Path(truth_manifest_path).resolve()
    if not truth.is_file():
        raise FileNotFoundError(f"PhyloLM truth manifest is missing: {truth}")
    expected_models = manifest_leaf_ids(truth)
    if set(population_specs) != set(expected_models):
        raise ValueError(
            f"{tree} population IDs must equal truth leaves; "
            f"missing={sorted(set(expected_models) - set(population_specs))}, "
            f"unexpected={sorted(set(population_specs) - set(expected_models))}"
        )
    populations: dict[str, list[dict[str, float]]] = {}
    pinned_populations: dict[str, dict[str, str]] = {}
    genome_pins: set[tuple[str, str, str]] = set()
    generation_contexts: set[tuple[str, str, str, str]] = set()
    for model_id in expected_models:
        spec = population_specs[model_id]
        if not isinstance(spec, Mapping):
            raise ValueError(f"{tree}/{model_id} population spec must be a mapping")
        path_text = spec.get("path")
        digest = spec.get("sha256")
        if not isinstance(path_text, str) or not _is_sha256(digest):
            raise ValueError(f"{tree}/{model_id} population spec requires path+sha256")
        path = Path(path_text).resolve()
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"{tree}/{model_id} population pin does not verify")
        payload = json.loads(path.read_text())
        _validate_population_payload(
            payload,
            suite_id=suite_id,
            tree_id=tree,
            model_id=model_id,
        )
        populations[model_id] = [dict(row) for row in payload["population"]]
        pinned_populations[model_id] = {"path": str(path), "sha256": digest}
        genome_pins.add(
            (
                str(payload["genome_receipt"]),
                str(payload["genome_receipt_sha256"]),
                str(payload["genes_sha256"]),
            )
        )
        generation = payload["generation_receipt"]
        generation_contexts.add(
            (
                str(generation["completion_receipt"]),
                str(generation["completion_receipt_sha256"]),
                str(generation["stage_manifest"]),
                str(generation["stage_manifest_sha256"]),
            )
        )
    if len(genome_pins) != 1:
        raise ValueError(f"{tree} populations do not use one identical sampled genome")
    genome_path, genome_digest, genes_digest = next(iter(genome_pins))
    genome = json.loads(Path(genome_path).read_text())
    _validate_genome_payload(genome)
    if sha256(genome_path) != genome_digest or genome["genes_sha256"] != genes_digest:
        raise ValueError(f"{tree} sampled-genome receipt does not verify")
    if len(generation_contexts) != 1:
        raise ValueError(f"{tree} populations do not share completion/stage provenance")
    completion_path, completion_digest, stage_path, stage_digest = next(
        iter(generation_contexts)
    )

    model_ids, similarity, distance = nei_distance_matrix(populations, protocol=protocol)
    if model_ids != expected_models:
        raise ValueError(f"{tree} PhyloLM model order differs from truth leaf order")
    newick = neighbor_joining_newick(model_ids, distance)
    truth_splits, truth_leaves = splits_from_manifest_path(str(truth))
    estimate_splits, estimate_leaves = splits_from_newick_text(newick)
    score = score_split_recovery(
        truth_splits,
        estimate_splits,
        truth_leaves,
        estimate_leaves,
    )
    if score["n_truth_splits"] <= 0:
        raise ValueError(f"{tree} is not topology eligible")

    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite PhyloLM tree output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        similarity_path = temporary / "phylolm_similarity.npy"
        distance_path = temporary / "phylolm_distance.npy"
        models_path = temporary / "models.json"
        tree_path = temporary / "phylolm_nj.newick"
        np.save(similarity_path, similarity)
        np.save(distance_path, distance)
        models_path.write_text(json.dumps(model_ids, indent=2) + "\n")
        tree_path.write_text(newick + "\n")
        receipt = {
            "schema": PHYLOLM_TREE_SCHEMA,
            "valid": True,
            "suite_id": suite_id,
            "tree_id": tree,
            "protocol": protocol.to_dict(),
            "truth_manifest": str(truth),
            "truth_sha256": sha256(truth),
            "genome_receipt": genome_path,
            "genome_receipt_sha256": genome_digest,
            "genes_sha256": genes_digest,
            "completion_receipt": {
                "path": completion_path,
                "sha256": completion_digest,
            },
            "stage_manifest": {"path": stage_path, "sha256": stage_digest},
            "population_receipts": pinned_populations,
            "model_ids": model_ids,
            "model_ids_sha256": canonical_sha256(model_ids),
            "outputs": {
                "similarity": {
                    "path": str(output / similarity_path.name),
                    "sha256": sha256(similarity_path),
                },
                "distance": {
                    "path": str(output / distance_path.name),
                    "sha256": sha256(distance_path),
                },
                "models": {
                    "path": str(output / models_path.name),
                    "sha256": sha256(models_path),
                },
                "tree": {
                    "path": str(output / tree_path.name),
                    "sha256": sha256(tree_path),
                },
            },
            "similarity_floor_hits": int(
                np.sum(similarity[np.triu_indices(len(model_ids), k=1)] <= protocol.similarity_floor)
            ),
            **score,
        }
        receipt_path = temporary / "receipt.json"
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, output)
        return receipt
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def build_phylolm_runset_receipt(
    tree_specs: Mapping[str, Mapping[str, str]],
    *,
    suite_id: str,
    completion_receipt: Mapping[str, str],
    stage_manifest: Mapping[str, str],
    protocol: PhyloLMProtocol = LOCKED_PHYLOLM_PROTOCOL,
) -> dict[str, Any]:
    """Validate exact common-46 tree receipts and build the Table 10 input."""

    _require_locked_protocol(protocol)
    _require_suite(suite_id)
    if set(tree_specs) != set(TOPOLOGY_TREE_IDS):
        raise ValueError("PhyloLM runset requires the exact canonical common-46 tree IDs")
    completion_pin = _verified_pin(completion_receipt, label="completion_receipt")
    validate_strict_training_completion(
        json.loads(Path(completion_pin["path"]).read_text()), cohort_id=suite_id
    )
    stage_pin = _verified_pin(stage_manifest, label="stage_manifest")
    rows = []
    truth_hashes: dict[str, str] = {}
    tree_receipts: dict[str, dict[str, str]] = {}
    genome_pins: set[tuple[str, str, str]] = set()
    for tree_id in TOPOLOGY_TREE_IDS:
        tree_pin = _verified_pin(tree_specs[tree_id], label=f"{tree_id} receipt")
        path = Path(tree_pin["path"])
        digest = tree_pin["sha256"]
        payload = json.loads(path.read_text())
        _validate_tree_receipt(payload, suite_id=suite_id, tree_id=tree_id)
        if payload.get("completion_receipt") != completion_pin:
            raise ValueError(f"{tree_id} PhyloLM completion receipt differs from runset")
        if payload.get("stage_manifest") != stage_pin:
            raise ValueError(f"{tree_id} PhyloLM stage manifest differs from runset")
        tree_receipts[tree_id] = {"path": str(path), "sha256": digest}
        truth_hashes[tree_id] = str(payload["truth_sha256"])
        genome_pins.add(
            (
                str(payload["genome_receipt"]),
                str(payload["genome_receipt_sha256"]),
                str(payload["genes_sha256"]),
            )
        )
        rows.append(
            {
                "tree_id": tree_id,
                "truth_sha256": payload["truth_sha256"],
                "tree_receipt": str(path),
                "tree_receipt_sha256": digest,
                **{
                    field: payload[field]
                    for field in (
                        "n_truth_splits",
                        "clade_recovery",
                        "polytomy_aware_exact_recovery",
                        "rf",
                        "false_negative",
                        "false_positive",
                    )
                },
            }
        )
    if len(genome_pins) != 1:
        raise ValueError("PhyloLM common-46 tree receipts use different sampled genomes")
    genome_path, genome_digest, genes_digest = next(iter(genome_pins))
    expected_group = f"runs_weighttraits_{suite_id}_fresh_phylolm"
    return {
        "schema": PHYLOLM_RECEIPT_SCHEMA,
        "valid": True,
        "suite_id": suite_id,
        "group": expected_group,
        "protocol": protocol.to_dict(),
        "completion_receipt": completion_pin,
        "stage_manifest": stage_pin,
        "genome_receipt": genome_path,
        "genome_receipt_sha256": genome_digest,
        "genes_sha256": genes_digest,
        "tree_receipts": tree_receipts,
        "common_tree_ids": list(TOPOLOGY_TREE_IDS),
        "n_trees": len(TOPOLOGY_TREE_IDS),
        "truth_manifest_sha256": truth_hashes,
        "truth_hashes_sha256": canonical_sha256(truth_hashes),
        "rows": rows,
    }


def write_phylolm_runset_receipt(payload: Mapping[str, Any], path: str | Path) -> None:
    validate_phylolm_runset_receipt(payload)
    _write_json_atomic(Path(path), payload)


def validate_phylolm_runset_receipt(
    payload: object, *, suite_id: str | None = None
) -> None:
    """Replay every native artifact behind an exact common-46 runset receipt."""

    if not isinstance(payload, Mapping):
        raise ValueError("PhyloLM runset receipt must be a JSON object")
    if payload.get("schema") != PHYLOLM_RECEIPT_SCHEMA or payload.get("valid") is not True:
        raise ValueError("PhyloLM runset receipt is not valid v1")
    observed_suite = str(payload.get("suite_id", ""))
    _require_suite(observed_suite)
    if suite_id is not None and observed_suite != suite_id:
        raise ValueError("PhyloLM runset receipt suite mismatch")
    if payload.get("group") != f"runs_weighttraits_{observed_suite}_fresh_phylolm":
        raise ValueError("PhyloLM runset group drift")
    if payload.get("protocol") != LOCKED_PHYLOLM_PROTOCOL.to_dict():
        raise ValueError("PhyloLM runset protocol drift")
    if payload.get("n_trees") != len(TOPOLOGY_TREE_IDS):
        raise ValueError("PhyloLM runset must declare n_trees=46")
    if payload.get("common_tree_ids") != list(TOPOLOGY_TREE_IDS):
        raise ValueError("PhyloLM runset common-tree IDs/order drift")
    completion_pin = _verified_pin(
        payload.get("completion_receipt"), label="PhyloLM runset completion receipt"
    )
    validate_strict_training_completion(
        json.loads(Path(completion_pin["path"]).read_text()),
        cohort_id=observed_suite,
    )
    stage_pin = _verified_pin(
        payload.get("stage_manifest"), label="PhyloLM runset stage manifest"
    )
    genome_path = Path(str(payload.get("genome_receipt", ""))).resolve()
    if (
        payload.get("genome_receipt") != str(genome_path)
        or not genome_path.is_file()
        or sha256(genome_path) != payload.get("genome_receipt_sha256")
    ):
        raise ValueError("PhyloLM runset genome pin does not verify")
    genome_payload = load_genome_receipt(genome_path)
    if payload.get("genes_sha256") != genome_payload.get("genes_sha256"):
        raise ValueError("PhyloLM runset sampled-gene hash drift")
    raw_tree_specs = payload.get("tree_receipts")
    if not isinstance(raw_tree_specs, Mapping) or set(raw_tree_specs) != set(
        TOPOLOGY_TREE_IDS
    ):
        raise ValueError("PhyloLM runset tree receipts are not exact common-46")
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != len(TOPOLOGY_TREE_IDS):
        raise ValueError("PhyloLM runset must contain exactly 46 rows")
    rows_by_tree: dict[str, Mapping[str, Any]] = {}
    for raw_row in raw_rows:
        if not isinstance(raw_row, Mapping):
            raise ValueError("PhyloLM runset rows must be mappings")
        tree_id = canonical_tree_id(raw_row.get("tree_id"))
        if tree_id in rows_by_tree:
            raise ValueError(f"PhyloLM runset duplicates {tree_id}")
        rows_by_tree[tree_id] = raw_row
    if set(rows_by_tree) != set(TOPOLOGY_TREE_IDS):
        raise ValueError("PhyloLM runset rows are not exact common-46")
    truth_hashes: dict[str, str] = {}
    for tree_id in TOPOLOGY_TREE_IDS:
        pin = _verified_pin(raw_tree_specs[tree_id], label=f"{tree_id} tree receipt")
        tree_payload = json.loads(Path(pin["path"]).read_text())
        _validate_tree_receipt(tree_payload, suite_id=observed_suite, tree_id=tree_id)
        if tree_payload.get("completion_receipt") != completion_pin:
            raise ValueError(f"{tree_id} completion provenance differs from runset")
        if tree_payload.get("stage_manifest") != stage_pin:
            raise ValueError(f"{tree_id} stage provenance differs from runset")
        if (
            tree_payload.get("genome_receipt") != str(genome_path)
            or tree_payload.get("genome_receipt_sha256")
            != payload.get("genome_receipt_sha256")
            or tree_payload.get("genes_sha256") != payload.get("genes_sha256")
        ):
            raise ValueError(f"{tree_id} sampled genome differs from runset")
        truth_hashes[tree_id] = str(tree_payload["truth_sha256"])
        row = rows_by_tree[tree_id]
        if (
            row.get("tree_receipt") != pin["path"]
            or row.get("tree_receipt_sha256") != pin["sha256"]
            or row.get("truth_sha256") != tree_payload.get("truth_sha256")
        ):
            raise ValueError(f"{tree_id} runset row provenance differs from tree receipt")
        for field in (
            "n_truth_splits",
            "clade_recovery",
            "polytomy_aware_exact_recovery",
            "rf",
            "false_negative",
            "false_positive",
        ):
            if row.get(field) != tree_payload.get(field):
                raise ValueError(f"{tree_id} runset row {field} differs from tree receipt")
    if payload.get("truth_manifest_sha256") != truth_hashes:
        raise ValueError("PhyloLM runset truth hash map differs from tree receipts")
    if payload.get("truth_hashes_sha256") != canonical_sha256(truth_hashes):
        raise ValueError("PhyloLM runset truth hash-set digest differs from replay")


def nei_similarity(
    left: Sequence[Mapping[str, float]], right: Sequence[Mapping[str, float]]
) -> float:
    if len(left) != len(right) or len(left) != LOCKED_PHYLOLM_PROTOCOL.n_genes:
        raise ValueError("PhyloLM populations must have the same exact 128 genes")
    numerator = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_gene, right_gene in zip(left, right, strict=True):
        for allele in set(left_gene) | set(right_gene):
            numerator += float(left_gene.get(allele, 0.0)) * float(
                right_gene.get(allele, 0.0)
            )
        left_norm += sum(float(value) ** 2 for value in left_gene.values())
        right_norm += sum(float(value) ** 2 for value in right_gene.values())
    denominator = math.sqrt(left_norm * right_norm)
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("PhyloLM population has invalid frequency norm")
    value = numerator / denominator
    if not math.isfinite(value) or value < 0 or value > 1 + 1e-12:
        raise ValueError("PhyloLM Nei similarity is outside [0,1]")
    return min(1.0, value)


def nei_distance_matrix(
    populations: Mapping[str, Sequence[Mapping[str, float]]],
    *,
    protocol: PhyloLMProtocol = LOCKED_PHYLOLM_PROTOCOL,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    _require_locked_protocol(protocol)
    if len(populations) < 4:
        raise ValueError("controlled-tree PhyloLM requires at least four leaf populations")
    model_ids = sorted(populations)
    for population in populations.values():
        _validate_population(population)
    similarity = np.eye(len(model_ids), dtype=np.float64)
    for left in range(len(model_ids)):
        for right in range(left + 1, len(model_ids)):
            value = nei_similarity(populations[model_ids[left]], populations[model_ids[right]])
            similarity[left, right] = similarity[right, left] = value
    distance = -np.log(np.maximum(similarity, protocol.similarity_floor))
    np.fill_diagonal(distance, 0.0)
    if not np.isfinite(distance).all() or not np.allclose(distance, distance.T):
        raise ValueError("PhyloLM distance matrix is not finite symmetric")
    return model_ids, similarity, distance


def _validate_genome_payload(payload: object) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("PhyloLM genome receipt must be a JSON object")
    if payload.get("schema") != PHYLOLM_GENOME_SCHEMA or payload.get("valid") is not True:
        raise ValueError("PhyloLM genome receipt is not valid v1")
    if payload.get("protocol") != LOCKED_PHYLOLM_PROTOCOL.to_dict():
        raise ValueError("PhyloLM genome receipt protocol drift")
    genes = payload.get("genes")
    indices = payload.get("selected_indices")
    if (
        not isinstance(genes, list)
        or len(genes) != LOCKED_PHYLOLM_PROTOCOL.n_genes
        or not all(isinstance(gene, str) and gene for gene in genes)
    ):
        raise ValueError("PhyloLM genome receipt must contain exact 128 nonempty genes")
    if (
        not isinstance(indices, list)
        or len(indices) != LOCKED_PHYLOLM_PROTOCOL.n_genes
        or len(set(indices)) != len(indices)
        or not all(isinstance(index, int) and not isinstance(index, bool) for index in indices)
    ):
        raise ValueError("PhyloLM genome receipt has invalid sampled indices")
    if payload.get("genes_sha256") != canonical_sha256(genes):
        raise ValueError("PhyloLM genome gene hash mismatch")
    if payload.get("selected_indices_sha256") != canonical_sha256(indices):
        raise ValueError("PhyloLM genome index hash mismatch")
    source = Path(str(payload.get("source_gene_pool", ""))).resolve()
    if not source.is_file() or sha256(source) != payload.get("source_gene_pool_sha256"):
        raise ValueError("PhyloLM source gene-pool pin does not verify")
    pool = load_gene_pool(source)
    if payload.get("source_gene_pool") != str(source):
        raise ValueError("PhyloLM source gene-pool path is not canonical")
    if payload.get("source_gene_count") != len(pool):
        raise ValueError("PhyloLM source gene count differs from the pinned pool")
    if any(index < 0 or index >= len(pool) for index in indices):
        raise ValueError("PhyloLM sampled index is outside the pinned gene pool")
    replay = np.random.default_rng(LOCKED_PHYLOLM_PROTOCOL.gene_seed).choice(
        len(pool),
        size=LOCKED_PHYLOLM_PROTOCOL.n_genes,
        replace=False,
    )
    expected_indices = [int(index) for index in replay]
    if indices != expected_indices:
        raise ValueError("PhyloLM sampled indices differ from seed-0 replay")
    if genes != [pool[index] for index in expected_indices]:
        raise ValueError("PhyloLM sampled genes differ from the pinned pool replay")


def _validate_population(population: Sequence[Mapping[str, float]]) -> None:
    if len(population) != LOCKED_PHYLOLM_PROTOCOL.n_genes:
        raise ValueError("PhyloLM population must contain exact 128 genes")
    for gene in population:
        if not isinstance(gene, Mapping) or not gene:
            raise ValueError("PhyloLM population gene must be a nonempty mapping")
        total = 0.0
        for allele, frequency in gene.items():
            if not isinstance(allele, str) or len(allele) > 4:
                raise ValueError("PhyloLM population has invalid allele")
            if isinstance(frequency, bool) or not isinstance(frequency, (int, float)):
                raise ValueError("PhyloLM population frequency must be numeric")
            value = float(frequency)
            scaled = value * LOCKED_PHYLOLM_PROTOCOL.samples_per_gene
            if not math.isfinite(value) or value <= 0 or not math.isclose(
                scaled, round(scaled), rel_tol=0.0, abs_tol=1e-10
            ):
                raise ValueError("PhyloLM population frequency violates 32-sample grid")
            total += value
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-10):
            raise ValueError("PhyloLM population gene frequencies do not sum to one")


def _validate_population_payload(
    payload: object,
    *,
    suite_id: str | None = None,
    tree_id: str | None = None,
    model_id: str | None = None,
) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("PhyloLM population receipt must be a JSON object")
    if payload.get("schema") != PHYLOLM_POPULATION_SCHEMA or payload.get("valid") is not True:
        raise ValueError("PhyloLM population receipt is not valid v1")
    if payload.get("protocol") != LOCKED_PHYLOLM_PROTOCOL.to_dict():
        raise ValueError("PhyloLM population receipt protocol drift")
    if suite_id is not None and payload.get("suite_id") != suite_id:
        raise ValueError("PhyloLM population receipt suite mismatch")
    if tree_id is not None and canonical_tree_id(payload.get("tree_id")) != tree_id:
        raise ValueError("PhyloLM population receipt tree mismatch")
    if model_id is not None and payload.get("model_id") != model_id:
        raise ValueError("PhyloLM population receipt model mismatch")
    population = payload.get("population")
    if not isinstance(population, list):
        raise ValueError("PhyloLM population receipt lacks population rows")
    _validate_population(population)
    if payload.get("population_sha256") != canonical_sha256(population):
        raise ValueError("PhyloLM population receipt hash mismatch")
    genome_path = Path(str(payload.get("genome_receipt", "")))
    if not genome_path.is_file() or sha256(genome_path) != payload.get(
        "genome_receipt_sha256"
    ):
        raise ValueError("PhyloLM population genome pin does not verify")
    genome = json.loads(genome_path.read_text())
    _validate_genome_payload(genome)
    if payload.get("genes_sha256") != genome.get("genes_sha256"):
        raise ValueError("PhyloLM population uses different sampled genes")
    if payload.get("generation_receipt_sha256") != canonical_sha256(
        payload.get("generation_receipt")
    ):
        raise ValueError("PhyloLM generation receipt hash mismatch")
    checkpoint = Path(str(payload.get("checkpoint", "")))
    if not checkpoint.exists() or not _is_sha256(payload.get("checkpoint_sha256")):
        raise ValueError("PhyloLM population checkpoint provenance is invalid")
    _validate_generation_receipt(
        payload.get("generation_receipt"),
        suite_id=str(payload.get("suite_id", "")),
        tree_id=canonical_tree_id(payload.get("tree_id")),
        model_id=str(payload.get("model_id", "")),
        checkpoint=checkpoint.resolve(),
        checkpoint_sha256=str(payload.get("checkpoint_sha256")),
    )


def _validate_generation_receipt(
    payload: object,
    *,
    suite_id: str,
    tree_id: str,
    model_id: str,
    checkpoint: Path,
    checkpoint_sha256: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("PhyloLM generation receipt must be a JSON object")
    if (
        payload.get("schema") != PHYLOLM_GENERATION_SCHEMA
        or payload.get("valid") is not True
    ):
        raise ValueError("PhyloLM generation receipt is not valid v1")
    if payload.get("protocol") != LOCKED_PHYLOLM_PROTOCOL.to_dict():
        raise ValueError("PhyloLM generation receipt protocol drift")
    if (
        payload.get("suite_id") != suite_id
        or canonical_tree_id(payload.get("tree_id")) != tree_id
        or payload.get("model_id") != model_id
    ):
        raise ValueError("PhyloLM generation receipt suite/tree/model drift")
    if (
        payload.get("batch_size") != LOCKED_PHYLOLM_PROTOCOL.batch_size
        or payload.get("torch_dtype") != LOCKED_PHYLOLM_PROTOCOL.torch_dtype
    ):
        raise ValueError("PhyloLM generation runtime differs from locked batch/dtype")
    if (
        payload.get("checkpoint") != str(checkpoint.resolve())
        or payload.get("checkpoint_sha256") != checkpoint_sha256
    ):
        raise ValueError("PhyloLM generation checkpoint provenance drift")
    for field in (
        "training_summary",
        "run_list",
        "ledger",
        "truth_manifest",
        "completion_receipt",
        "stage_manifest",
    ):
        _validate_named_file_pin(payload, field, label=f"PhyloLM generation {field}")
    completion = json.loads(Path(str(payload["completion_receipt"])).read_text())
    validate_strict_training_completion(completion, cohort_id=suite_id)
    truth = Path(str(payload["truth_manifest"]))
    if model_id not in manifest_leaf_ids(truth):
        raise ValueError("PhyloLM generation model is not a truth-manifest leaf")
    loader = payload.get("loader")
    if not isinstance(loader, Mapping):
        raise ValueError("PhyloLM generation receipt lacks loader provenance")
    required_loader = {
        "backend": "huggingface_text_generation_pipeline",
        "checkpoint": str(checkpoint.resolve()),
        "local_files_only": True,
        "torch_dtype": LOCKED_PHYLOLM_PROTOCOL.torch_dtype,
    }
    for field, expected in required_loader.items():
        if loader.get(field) != expected:
            raise ValueError(f"PhyloLM generation loader {field} drift")
    if loader.get("checkpoint_artifact") not in {"model", "merged"}:
        raise ValueError("PhyloLM generation loader has invalid checkpoint artifact")
    for field in ("tokenizer_model_id", "tokenizer_revision"):
        if not isinstance(loader.get(field), str) or not str(loader[field]).strip():
            raise ValueError(f"PhyloLM generation loader lacks {field}")
    for field in ("torch_version", "transformers_version"):
        if not isinstance(loader.get(field), str) or not str(loader[field]).strip():
            raise ValueError(f"PhyloLM generation loader lacks {field}")


def _validate_named_file_pin(payload: Mapping[str, Any], field: str, *, label: str) -> None:
    path = Path(str(payload.get(field, ""))).resolve()
    digest = payload.get(f"{field}_sha256")
    if (
        payload.get(field) != str(path)
        or not path.is_file()
        or not _is_sha256(digest)
        or sha256(path) != digest
    ):
        raise ValueError(f"{label} pin does not verify")


def _validate_tree_receipt(
    payload: object, *, suite_id: str, tree_id: str
) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("PhyloLM tree receipt must be a JSON object")
    if payload.get("schema") != PHYLOLM_TREE_SCHEMA or payload.get("valid") is not True:
        raise ValueError("PhyloLM tree receipt is not valid v1")
    if payload.get("suite_id") != suite_id or canonical_tree_id(
        payload.get("tree_id")
    ) != tree_id:
        raise ValueError("PhyloLM tree receipt suite/tree mismatch")
    if payload.get("protocol") != LOCKED_PHYLOLM_PROTOCOL.to_dict():
        raise ValueError("PhyloLM tree receipt protocol drift")
    truth = Path(str(payload.get("truth_manifest", "")))
    if not truth.is_file() or sha256(truth) != payload.get("truth_sha256"):
        raise ValueError("PhyloLM tree receipt truth pin does not verify")
    genome = Path(str(payload.get("genome_receipt", "")))
    if not genome.is_file() or sha256(genome) != payload.get("genome_receipt_sha256"):
        raise ValueError("PhyloLM tree receipt genome pin does not verify")
    genome_payload = json.loads(genome.read_text())
    _validate_genome_payload(genome_payload)
    if payload.get("genes_sha256") != genome_payload.get("genes_sha256"):
        raise ValueError("PhyloLM tree receipt sampled-gene hash mismatch")
    model_ids = payload.get("model_ids")
    if (
        not isinstance(model_ids, list)
        or len(model_ids) < 4
        or model_ids != sorted(model_ids)
        or len(model_ids) != len(set(model_ids))
        or not all(isinstance(model_id, str) and model_id for model_id in model_ids)
    ):
        raise ValueError("PhyloLM tree receipt has invalid model IDs")
    if model_ids != manifest_leaf_ids(truth):
        raise ValueError("PhyloLM tree receipt model IDs differ from truth leaves")
    if payload.get("model_ids_sha256") != canonical_sha256(model_ids):
        raise ValueError("PhyloLM tree receipt model-ID hash mismatch")
    population_specs = payload.get("population_receipts")
    if not isinstance(population_specs, Mapping) or set(population_specs) != set(model_ids):
        raise ValueError("PhyloLM tree receipt population pins differ from model IDs")
    populations: dict[str, list[dict[str, float]]] = {}
    generation_contexts: set[tuple[str, str, str, str]] = set()
    for model_id in model_ids:
        population_pin = _verified_pin(
            population_specs[model_id], label=f"{tree_id}/{model_id} population"
        )
        population_payload = json.loads(Path(population_pin["path"]).read_text())
        _validate_population_payload(
            population_payload,
            suite_id=suite_id,
            tree_id=tree_id,
            model_id=model_id,
        )
        if (
            population_payload.get("genome_receipt_sha256")
            != payload.get("genome_receipt_sha256")
            or population_payload.get("genes_sha256") != payload.get("genes_sha256")
        ):
            raise ValueError("PhyloLM tree population uses different sampled genes")
        populations[model_id] = [dict(row) for row in population_payload["population"]]
        generation = population_payload["generation_receipt"]
        generation_contexts.add(
            (
                str(generation["completion_receipt"]),
                str(generation["completion_receipt_sha256"]),
                str(generation["stage_manifest"]),
                str(generation["stage_manifest_sha256"]),
            )
        )
    if len(generation_contexts) != 1:
        raise ValueError("PhyloLM tree populations mix completion/stage provenance")
    completion_path, completion_digest, stage_path, stage_digest = next(
        iter(generation_contexts)
    )
    if payload.get("completion_receipt") != {
        "path": completion_path,
        "sha256": completion_digest,
    }:
        raise ValueError("PhyloLM tree completion provenance differs from populations")
    if payload.get("stage_manifest") != {
        "path": stage_path,
        "sha256": stage_digest,
    }:
        raise ValueError("PhyloLM tree stage provenance differs from populations")
    outputs = payload.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != {
        "similarity",
        "distance",
        "models",
        "tree",
    }:
        raise ValueError("PhyloLM tree receipt output pins are incomplete")
    output_pins = {
        name: _verified_pin(spec, label=f"{tree_id} {name}")
        for name, spec in outputs.items()
    }
    recorded_models = json.loads(Path(output_pins["models"]["path"]).read_text())
    if recorded_models != model_ids:
        raise ValueError("PhyloLM tree models artifact differs from receipt")
    similarity = np.asarray(
        np.load(output_pins["similarity"]["path"], allow_pickle=False),
        dtype=np.float64,
    )
    distance = np.asarray(
        np.load(output_pins["distance"]["path"], allow_pickle=False),
        dtype=np.float64,
    )
    shape = (len(model_ids), len(model_ids))
    if similarity.shape != shape or distance.shape != shape:
        raise ValueError("PhyloLM tree matrices have wrong shape")
    if (
        not np.isfinite(similarity).all()
        or not np.isfinite(distance).all()
        or not np.allclose(similarity, similarity.T, rtol=0.0, atol=1e-12)
        or not np.allclose(distance, distance.T, rtol=0.0, atol=1e-12)
        or not np.allclose(np.diag(similarity), 1.0, rtol=0.0, atol=1e-12)
        or not np.allclose(np.diag(distance), 0.0, rtol=0.0, atol=1e-12)
        or np.any(similarity < -1e-12)
        or np.any(similarity > 1 + 1e-12)
        or np.any(distance < -1e-12)
    ):
        raise ValueError("PhyloLM tree matrices violate distance/similarity invariants")
    expected_distance = -np.log(
        np.maximum(similarity, LOCKED_PHYLOLM_PROTOCOL.similarity_floor)
    )
    np.fill_diagonal(expected_distance, 0.0)
    if not np.allclose(distance, expected_distance, rtol=0.0, atol=1e-12):
        raise ValueError("PhyloLM distance is not the locked Nei transform")
    replay_model_ids, replay_similarity, replay_distance = nei_distance_matrix(
        populations
    )
    if replay_model_ids != model_ids:
        raise ValueError("PhyloLM tree model order differs from population replay")
    if not np.allclose(similarity, replay_similarity, rtol=0.0, atol=1e-12):
        raise ValueError("PhyloLM similarity matrix differs from population replay")
    if not np.allclose(distance, replay_distance, rtol=0.0, atol=1e-12):
        raise ValueError("PhyloLM distance matrix differs from population replay")
    floor_hits = int(
        np.sum(
            replay_similarity[np.triu_indices(len(model_ids), k=1)]
            <= LOCKED_PHYLOLM_PROTOCOL.similarity_floor
        )
    )
    if payload.get("similarity_floor_hits") != floor_hits:
        raise ValueError("PhyloLM similarity-floor count differs from replay")
    newick = Path(output_pins["tree"]["path"]).read_text().strip()
    replay_newick = neighbor_joining_newick(model_ids, replay_distance).strip()
    if newick != replay_newick:
        raise ValueError("PhyloLM NJ tree differs from locked distance replay")
    truth_splits, truth_leaves = splits_from_manifest_path(str(truth))
    estimate_splits, estimate_leaves = splits_from_newick_text(newick)
    if estimate_leaves != set(model_ids):
        raise ValueError("PhyloLM NJ tree leaves differ from model IDs")
    recomputed = score_split_recovery(
        truth_splits,
        estimate_splits,
        truth_leaves,
        estimate_leaves,
    )
    for field in (
        "n_truth_splits",
        "rf",
        "false_negative",
        "false_positive",
    ):
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"PhyloLM tree receipt has invalid {field}")
    clade = payload.get("clade_recovery")
    paer = payload.get("polytomy_aware_exact_recovery")
    if not isinstance(clade, (int, float)) or isinstance(clade, bool) or not 0 <= clade <= 1:
        raise ValueError("PhyloLM tree receipt has invalid clade_recovery")
    if not isinstance(paer, bool):
        raise ValueError("PhyloLM tree receipt has invalid PAER")
    n_truth = payload["n_truth_splits"]
    false_negative = payload["false_negative"]
    if n_truth <= 0 or not math.isclose(
        float(clade), (n_truth - false_negative) / n_truth, abs_tol=1e-12
    ):
        raise ValueError("PhyloLM tree receipt clade/FN counts are inconsistent")
    if payload["rf"] != false_negative + payload["false_positive"]:
        raise ValueError("PhyloLM tree receipt RF is not FN+FP")
    if paer != (false_negative == 0):
        raise ValueError("PhyloLM tree receipt PAER is inconsistent with FN")
    for field in (
        "n_truth_splits",
        "clade_recovery",
        "polytomy_aware_exact_recovery",
        "rf",
        "false_negative",
        "false_positive",
    ):
        if payload.get(field) != recomputed[field]:
            raise ValueError(f"PhyloLM tree receipt {field} differs from recomputation")


def _verified_pin(raw: Mapping[str, str], *, label: str) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must be a path+sha256 mapping")
    path = Path(str(raw.get("path", ""))).resolve()
    digest = raw.get("sha256")
    if not path.is_file() or not _is_sha256(digest) or sha256(path) != digest:
        raise ValueError(f"{label} pin does not verify")
    return {"path": str(path), "sha256": str(digest)}


def _require_locked_protocol(protocol: PhyloLMProtocol) -> None:
    if protocol != LOCKED_PHYLOLM_PROTOCOL:
        raise ValueError("corrected Table 10 requires the exact locked PhyloLM protocol")


def _require_suite(suite_id: str) -> None:
    if suite_id not in SUITE_CONTRACTS:
        raise ValueError(f"unsupported corrected PhyloLM suite: {suite_id!r}")


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    return value.strip()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    output = path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
