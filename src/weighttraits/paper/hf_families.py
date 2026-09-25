"""Documented Hugging Face families (App. D): clades estimated and tree-likeness per method.

For each family and each method (weights: BHV Frechet mean of per-layer cosine NJ trees;
PhyloLM: NJ on its behavioral distance) this scores the estimate against the documented tree
(false negatives over the documented clades) and computes the label-free tree-likeness score
``A`` over all quartets of the method's distance matrix. The weight matrix is the layer-averaged
cosine distance; the PhyloLM matrix is ``-log max(S, 1e-3)`` of its Nei similarity.

``per_layer_clade_support`` gives the fraction of per-layer NJ trees that contain each
documented clade (App. D reports 95.2% for the Llama-2 chat clade); it needs the per-layer cube.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from weighttraits.phylo.additivity import four_point_additivity
from weighttraits.phylo.reconstruct import neighbor_joining_newick
from weighttraits.phylo.splits import Split, splits_from_newick_text


@dataclass(frozen=True)
class MethodInputs:
    estimate_newick: Path
    distance_matrix: Path  # (n, n) .npy
    labels: Sequence[str] | Path  # label list, or a distance_metadata.json with model_names


def _labels(value: Sequence[str] | Path) -> list[str]:
    if isinstance(value, Path):
        metadata = json.loads(value.read_text())
        return [str(name) for name in metadata["model_names"]]
    return [str(name) for name in value]


def score_method(truth: tuple[set[Split], set[str]], inputs: MethodInputs) -> dict[str, Any]:
    truth_splits, truth_leaves = truth
    estimate_splits, estimate_leaves = splits_from_newick_text(inputs.estimate_newick.read_text())
    if estimate_leaves != truth_leaves:
        raise ValueError(f"{inputs.estimate_newick}: leaves differ from the documented tree")
    labels = _labels(inputs.labels)
    if set(labels) != truth_leaves:
        raise ValueError(f"{inputs.distance_matrix}: labels differ from the documented tree")
    matrix = np.abs(np.load(inputs.distance_matrix))
    additivity = four_point_additivity(labels, matrix, truth_splits=truth_splits, max_quartets=None)
    missed = truth_splits - estimate_splits
    return {
        "n_documented_clades": len(truth_splits),
        "false_negatives": len(missed),
        "clades_estimated": len(truth_splits) - len(missed),
        "missed_clades": sorted(sorted(split) for split in missed),
        "tree_likeness_all_quartets": additivity.all.mean_additivity,
        "tree_likeness_informative_quartets": (
            additivity.informative.mean_additivity if additivity.informative else None
        ),
        "n_quartets": additivity.n_quartets_scored,
    }


def score_family(truth_newick: Path, methods: Mapping[str, MethodInputs]) -> dict[str, Any]:
    truth = splits_from_newick_text(Path(truth_newick).read_text())
    return {
        "truth_newick": str(truth_newick),
        "n_leaves": len(truth[1]),
        "methods": {name: score_method(truth, inputs) for name, inputs in methods.items()},
    }


def per_layer_clade_support(
    cube: np.ndarray, labels: Sequence[str], truth_newick: Path, *, zero_tolerance: float = 1e-12
) -> dict[str, Any]:
    """Fraction of per-layer NJ trees that contain each documented clade.

    Layers whose distances are all numerically zero are skipped (they never changed).
    """

    truth_splits, truth_leaves = splits_from_newick_text(Path(truth_newick).read_text())
    if set(labels) != truth_leaves:
        raise ValueError("cube labels differ from the documented tree")
    cube = np.abs(np.asarray(cube, dtype=float))
    active = [i for i in range(cube.shape[0]) if np.max(cube[i]) > zero_tolerance]
    hits = {split: 0 for split in truth_splits}
    for index in active:
        splits, _ = splits_from_newick_text(neighbor_joining_newick(list(labels), cube[index]))
        for split in truth_splits:
            hits[split] += split in splits
    return {
        "n_layers": int(cube.shape[0]),
        "n_active_layers": len(active),
        "clade_support": [
            {"clade": sorted(split), "fraction_of_layer_trees": hits[split] / len(active)}
            for split in sorted(truth_splits, key=lambda s: (len(s), sorted(s)))
        ],
    }
