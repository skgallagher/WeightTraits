"""Faithful model-independent PhyloLM population and distance implementation."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from weighttraits.distances.streaming import DistanceCube, write_distance_cube
from weighttraits.phylo.reconstruct import neighbor_joining_newick


PHYLOLM_UPSTREAM_COMMIT = "8c70edf062a0adce2a3e6c8c79cd23a645fd0905"
DEFAULT_GENES = 128
DEFAULT_SAMPLES_PER_GENE = 32
DEFAULT_NEW_TOKENS = 4
DEFAULT_ALLELE_CHARACTERS = 4


def load_gene_pool(path: str | Path) -> list[str]:
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, list) or not raw or not all(isinstance(item, str) for item in raw):
        raise ValueError("PhyloLM gene pool must be a non-empty JSON list of strings")
    if any(not item for item in raw):
        raise ValueError("PhyloLM gene pool contains an empty gene")
    return raw


def sample_genes(pool: Sequence[str], *, n_genes: int = DEFAULT_GENES, seed: int = 0) -> list[str]:
    genes = list(pool)
    if n_genes <= 0:
        raise ValueError("n_genes must be positive")
    if n_genes > len(genes):
        raise ValueError(f"n_genes={n_genes} exceeds gene pool size {len(genes)}")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(genes), size=n_genes, replace=False)
    return [genes[int(index)] for index in indices]


def write_sampled_genome(
    path: str | Path,
    *,
    genes: Sequence[str],
    source_path: str | Path,
    seed: int,
) -> None:
    source = Path(source_path)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "method": "phylolm",
                "upstream_commit": PHYLOLM_UPSTREAM_COMMIT,
                "source_gene_pool": str(source),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "seed": seed,
                "n_genes": len(genes),
                "genes": list(genes),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def compute_population(alleles: Sequence[Sequence[str]]) -> list[dict[str, float]]:
    """Convert genes x samples allele strings into per-gene frequency dictionaries."""

    rows = [list(row) for row in alleles]
    if not rows or not rows[0]:
        raise ValueError("PhyloLM alleles must be a non-empty genes x samples matrix")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("PhyloLM allele rows must have equal sample counts")
    populations: list[dict[str, float]] = []
    for row in rows:
        if not all(isinstance(allele, str) for allele in row):
            raise ValueError("PhyloLM alleles must be strings")
        counts = Counter(row)
        populations.append({allele: count / width for allele, count in sorted(counts.items())})
    return populations


def nei_similarity(
    left: Sequence[dict[str, float]], right: Sequence[dict[str, float]]
) -> float:
    """Compute the PhyloLM Nei similarity from two population matrices."""

    if len(left) != len(right) or not left:
        raise ValueError("PhyloLM populations must have the same non-zero gene count")
    numerator = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_gene, right_gene in zip(left, right):
        for allele in set(left_gene) | set(right_gene):
            numerator += left_gene.get(allele, 0.0) * right_gene.get(allele, 0.0)
        left_norm += sum(value * value for value in left_gene.values())
        right_norm += sum(value * value for value in right_gene.values())
    denominator = math.sqrt(left_norm * right_norm)
    if denominator <= 0:
        raise ValueError("PhyloLM population has zero frequency norm")
    return numerator / denominator


def nei_distance_matrix(
    populations: dict[str, Sequence[dict[str, float]]], *, eps: float = 1e-3
) -> tuple[list[str], np.ndarray, np.ndarray]:
    if len(populations) < 2:
        raise ValueError("at least two PhyloLM populations are required")
    if not 0 < eps < 1:
        raise ValueError("eps must be between zero and one")
    model_ids = sorted(populations)
    gene_counts = {len(populations[model_id]) for model_id in model_ids}
    if len(gene_counts) != 1 or next(iter(gene_counts)) <= 0:
        raise ValueError("all PhyloLM populations must have the same non-zero gene count")
    similarity = np.eye(len(model_ids), dtype=np.float64)
    for left in range(len(model_ids)):
        for right in range(left + 1, len(model_ids)):
            value = nei_similarity(populations[model_ids[left]], populations[model_ids[right]])
            similarity[left, right] = similarity[right, left] = value
    distance = -np.log(np.maximum(similarity, eps))
    np.fill_diagonal(distance, 0.0)
    return model_ids, similarity, distance


def save_population(
    path: str | Path,
    *,
    model_id: str,
    population: Sequence[dict[str, float]],
    metadata: dict[str, Any] | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "method": "phylolm",
                "upstream_commit": PHYLOLM_UPSTREAM_COMMIT,
                "model_id": model_id,
                "population": list(population),
                "metadata": metadata or {},
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def load_population_directory(path: str | Path) -> tuple[dict[str, list[dict[str, float]]], dict[str, Any]]:
    populations: dict[str, list[dict[str, float]]] = {}
    metadata: dict[str, Any] = {}
    files = sorted(Path(path).glob("*.json"))
    if not files:
        raise ValueError(f"no PhyloLM population JSON files found in {path}")
    for file in files:
        row = json.loads(file.read_text())
        model_id = str(row["model_id"])
        if model_id in populations:
            raise ValueError(f"duplicate PhyloLM population model ID: {model_id}")
        population = row.get("population", row.get("P"))
        if not isinstance(population, list):
            raise ValueError(f"invalid PhyloLM population in {file}")
        populations[model_id] = [dict(item) for item in population]
        metadata[model_id] = dict(row.get("metadata", row.get("meta", {})))
    return populations, metadata


def write_phylolm_analysis(
    population_dir: str | Path,
    out_dir: str | Path,
    *,
    eps: float = 1e-3,
) -> dict[str, Any]:
    populations, per_model_metadata = load_population_directory(population_dir)
    model_ids, similarity, distance = nei_distance_matrix(populations, eps=eps)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "phylolm_similarity.npy", similarity)
    np.save(output / "phylolm_distance.npy", distance)
    with (output / "phylolm_distance.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model_id", *model_ids])
        for index, model_id in enumerate(model_ids):
            writer.writerow([model_id, *[float(value) for value in distance[index]]])
    cube = DistanceCube(
        distances={"phylolm": distance[None, :, :]},
        layer_names=["phylolm_nei"],
        model_ids=model_ids,
        audit={
            "schema_version": 1,
            "representation": "phylolm_allele_population",
            "upstream_commit": PHYLOLM_UPSTREAM_COMMIT,
            "distance": f"-log(max(nei_similarity, {eps}))",
            "n_models": len(model_ids),
            "n_genes": len(next(iter(populations.values()))),
            "source_population_dir": str(population_dir),
            "per_model_metadata": per_model_metadata,
        },
    )
    cube_dir = output / "distance_cube"
    write_distance_cube(cube, cube_dir)
    tree = neighbor_joining_newick(model_ids, distance)
    (output / "phylolm_nj.newick").write_text(tree + "\n")
    audit = {
        **cube.audit,
        "cube_dir": str(cube_dir),
        "tree_path": str(output / "phylolm_nj.newick"),
        "similarity_floor_hits": int(np.sum(similarity <= eps)),
    }
    (output / "phylolm_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return audit
