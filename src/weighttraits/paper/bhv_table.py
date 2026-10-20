"""All-weight estimates for Table 2: BHV Frechet means of per-layer neighbor-joining trees.

For each controlled tree, every trained layer's cosine-distance matrix gives one NJ tree.
Layers whose distances are all (numerically) zero never changed during training and carry no
lineage information, so they are excluded. The remaining layer trees are averaged in BHV tree
space (Sturm's proximal-point Frechet mean) and the mean tree is scored against the truth.

Ported from ELLMTrees ``scripts/run_controlled_bhv.py``, which produced the paper's Table 2
estimation columns and the RF table (scale mode ``raw``, seed 0, 300 epochs).

Two scale modes are supported: ``raw`` uses the cosine distances as branch lengths;
``mean_pairwise`` divides each layer's matrix by its mean off-diagonal distance first, so every
layer enters tree space on a common scale.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from weighttraits.analysis.direct import truth_newick_from_manifest
from weighttraits.phylo import bhv
from weighttraits.phylo.reconstruct import neighbor_joining_newick
from weighttraits.phylo.recovery import aggregate_recovery, score_split_recovery
from weighttraits.phylo.splits import splits_from_newick_text

SCHEMA = "weighttraits.bhv_table.v1"
ALL_TREE_IDS = tuple(f"confirm_paper_tree_{index:03d}" for index in range(1, 51))
NO_CLADE_TREE_IDS = frozenset(f"confirm_paper_tree_{index:03d}" for index in (15, 20, 37, 47))
TOPOLOGY_TREE_IDS = tuple(tree_id for tree_id in ALL_TREE_IDS if tree_id not in NO_CLADE_TREE_IDS)
ANALYSIS_DIR_NAMES = ("model_leaf_analysis", "merged_leaf_analysis", "cumulative_leaf_analysis")
SCALE_MODES = ("raw", "mean_pairwise")
SUMMARY_COLUMNS = (
    "suite_id",
    "metric",
    "scale_mode",
    "seed",
    "epochs",
    "n_trees",
    "n_seed_stable_complete_split_sets",
    "seed_stable_complete_split_fraction",
    "n_seed_stable_truth_recovery",
    "seed_stable_truth_recovery_fraction",
    "clade_recovery_mean",
    "clade_recovery_se",
    "polytomy_aware_exact_recovery_rate",
    "polytomy_aware_exact_recovery_rate_se",
    "rf_mean",
    "rf_se",
    "false_negative_mean",
    "false_negative_se",
    "pooled_clade_recovery",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analysis_dir(root: Path, tree_id: str) -> Path:
    """The single per-tree analysis directory (full model, merged LoRA, or cumulative LoRA)."""

    tree_root = root / tree_id
    matches = [tree_root / name for name in ANALYSIS_DIR_NAMES if (tree_root / name).is_dir()]
    if len(matches) != 1:
        raise ValueError(
            f"{tree_root} must contain exactly one recognized analysis directory; "
            f"found {[path.name for path in matches]}"
        )
    return matches[0]


def active_layer_indices(cube: np.ndarray, *, zero_tolerance: float) -> list[int]:
    """Layers with at least one pairwise distance above ``zero_tolerance``."""

    if cube.ndim != 3 or cube.shape[1] != cube.shape[2]:
        raise ValueError(f"distance cube must be layer x model x model, got {cube.shape}")
    if not np.all(np.isfinite(cube)):
        raise ValueError("distance cube contains non-finite values")
    maxima = np.max(np.abs(cube), axis=(1, 2))
    return [int(index) for index in np.flatnonzero(maxima > zero_tolerance)]


def scaled_matrix(matrix: np.ndarray, mode: str) -> tuple[np.ndarray, float]:
    values = np.asarray(matrix, dtype=np.float64)
    if mode == "raw":
        return values, 1.0
    if mode != "mean_pairwise":
        raise ValueError(f"unknown scale mode: {mode}")
    off_diagonal = values[np.triu_indices(values.shape[0], 1)]
    scale = float(np.mean(off_diagonal))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"active layer has invalid mean pairwise distance {scale}")
    return values / scale, scale


def layer_trees(
    cube: np.ndarray,
    labels: list[str],
    indices: list[int],
    *,
    mode: str,
    branch_length_format: str | None = None,
) -> tuple[list[bhv.BHVTree], int, list[float]]:
    """One NJ tree per active layer; negative pendant edges from NJ are clipped to zero.

    ``branch_length_format=None`` reproduces the paper (five-decimal Newick, which zeroes edges
    shorter than 5e-6); ``"%.17g"`` keeps full precision.
    """

    trees: list[bhv.BHVTree] = []
    clipped_pendant_edges = 0
    scales: list[float] = []
    for index in indices:
        matrix, scale = scaled_matrix(cube[index], mode)
        tree = bhv.from_newick(
            neighbor_joining_newick(labels, matrix, branch_length_format=branch_length_format)
        )
        clipped_pendant_edges += sum(length < 0.0 for length in tree.pendant.values())
        tree.pendant = {label: max(0.0, float(length)) for label, length in tree.pendant.items()}
        trees.append(tree)
        scales.append(scale)
    return trees, clipped_pendant_edges, scales


def score_mean(
    mean: bhv.BHVTree, *, truth_splits: set[frozenset[str]], truth_leaves: set[str]
) -> tuple[dict[str, Any], str, list[list[str]]]:
    newick = bhv.to_newick(mean)
    estimate_splits, estimate_leaves = splits_from_newick_text(newick)
    score = score_split_recovery(truth_splits, estimate_splits, truth_leaves, estimate_leaves)
    return score, newick, sorted(sorted(split) for split in estimate_splits)


def load_tree_cube(
    root: Path, tree_id: str, metric: str, *, zero_tolerance: float
) -> tuple[np.ndarray, list[str], list[str], list[dict[str, str]], int, float]:
    """Load one tree's distance cube, clipping numerically negative distances to zero."""

    current = analysis_dir(root, tree_id)
    paths = [current / name for name in ("direct_distance_layers.npz", "layers.json", "models.json")]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    inputs = [{"path": str(path), "sha256": sha256(path)} for path in paths]
    layer_names = [str(value) for value in json.loads(paths[1].read_text())]
    labels = [str(value) for value in json.loads(paths[2].read_text())]
    with np.load(paths[0], allow_pickle=False) as archive:
        if metric not in archive.files:
            raise ValueError(f"{paths[0]} lacks metric {metric!r}")
        cube = np.asarray(archive[metric], dtype=np.float64)
    if cube.shape != (len(layer_names), len(labels), len(labels)):
        raise ValueError(
            f"{tree_id} cube shape {cube.shape} does not match "
            f"{len(layer_names)} layers and {len(labels)} labels"
        )
    minimum = float(np.min(cube))
    if minimum < -zero_tolerance:
        raise ValueError(
            f"{tree_id} contains a distance {minimum} below the numerical tolerance "
            f"{-zero_tolerance}"
        )
    n_negative = int(np.count_nonzero(cube < 0.0))
    if n_negative:
        cube = np.maximum(cube, 0.0)
    return cube, layer_names, labels, inputs, n_negative, minimum


def run_bhv_table(
    suites: Sequence[tuple[str, Path]],
    *,
    truth_root: Path,
    out_dir: Path,
    metric: str = "cosine",
    scale_modes: Sequence[str] = ("raw",),
    seeds: Sequence[int] = (0,),
    epochs: int = 300,
    zero_tolerance: float = 1e-12,
    tree_ids: Sequence[str] = TOPOLOGY_TREE_IDS,
    expected_active: Mapping[str, int] | None = None,
    branch_length_format: str | None = None,
    log=print,
) -> dict[str, Any]:
    """Compute BHV all-weight estimates for each suite and write JSON + summary CSV."""

    unknown = sorted(set(scale_modes) - set(SCALE_MODES))
    if unknown:
        raise ValueError(f"unknown scale modes: {unknown}")
    expected_active = dict(expected_active or {})
    out_dir = Path(out_dir)
    trees_out = out_dir / "trees"
    trees_out.mkdir(parents=True, exist_ok=True)
    observations: list[dict[str, Any]] = []
    suite_audits: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for suite_id, root in suites:
        root = Path(root)
        log(f"[{suite_id}] {root}")
        canonical_active: list[str] | None = None
        canonical_total: int | None = None
        n_clipped_entries = 0
        minimum_before_clipping = 0.0
        input_files: list[dict[str, str]] = []
        records: dict[tuple[str, int], list[dict[str, Any]]] = {
            (mode, seed): [] for mode in scale_modes for seed in seeds
        }
        stable_splits = {mode: 0 for mode in scale_modes}
        stable_truth = {mode: 0 for mode in scale_modes}

        for position, tree_id in enumerate(tree_ids, start=1):
            cube, layer_names, labels, inputs, n_negative, minimum = load_tree_cube(
                root, tree_id, metric, zero_tolerance=zero_tolerance
            )
            input_files.extend(inputs)
            n_clipped_entries += n_negative
            minimum_before_clipping = min(minimum_before_clipping, minimum)
            indices = active_layer_indices(cube, zero_tolerance=zero_tolerance)
            active_names = [layer_names[index] for index in indices]
            if not active_names:
                raise ValueError(f"{suite_id}/{tree_id} has no active layers")
            if canonical_active is None:
                canonical_active, canonical_total = active_names, len(layer_names)
                expected = expected_active.get(suite_id)
                if expected is not None and len(active_names) != expected:
                    raise ValueError(
                        f"{suite_id} expected {expected} active layers, found {len(active_names)}"
                    )
            elif active_names != canonical_active or len(layer_names) != canonical_total:
                raise ValueError(f"{suite_id}/{tree_id} active layers differ from the suite")

            manifest_path = Path(truth_root) / f"{tree_id}.manifest.jsonl"
            if not manifest_path.is_file():
                raise FileNotFoundError(manifest_path)
            truth_splits, truth_leaves = splits_from_newick_text(
                truth_newick_from_manifest(manifest_path)
            )
            if not truth_splits:
                raise ValueError(f"{tree_id} has no clade to estimate")
            log(f"  {position:02d}/{len(tree_ids)} {tree_id}: {len(indices)}/{len(layer_names)} active")

            for mode in scale_modes:
                trees, n_clipped, scales = layer_trees(
                    cube, labels, indices, mode=mode, branch_length_format=branch_length_format
                )
                seed_splits: list[tuple[tuple[str, ...], ...]] = []
                seed_truth: list[tuple[int, int]] = []
                for seed in seeds:
                    mean = bhv.frechet_mean(trees, n_epochs=epochs, seed=seed, tol=0.0)
                    score, newick, splits = score_mean(
                        mean, truth_splits=truth_splits, truth_leaves=truth_leaves
                    )
                    seed_splits.append(tuple(tuple(split) for split in splits))
                    seed_truth.append((int(score["true_positive"]), int(score["false_negative"])))
                    tree_path = trees_out / suite_id / mode / tree_id / f"seed_{seed}.newick"
                    tree_path.parent.mkdir(parents=True, exist_ok=True)
                    tree_path.write_text(newick.strip() + "\n")
                    observations.append(
                        {
                            "suite_id": suite_id,
                            "tree_id": tree_id,
                            "metric": metric,
                            "scale_mode": mode,
                            "seed": seed,
                            "epochs": epochs,
                            "n_models": len(labels),
                            "n_tensors_total": len(layer_names),
                            "n_tensors_active": len(indices),
                            "n_tensors_zero_distance": len(layer_names) - len(indices),
                            "n_negative_pendant_edges_clipped": n_clipped,
                            "tensor_scale_min": min(scales),
                            "tensor_scale_max": max(scales),
                            "tensor_scale_mean": statistics.fmean(scales),
                            "estimate_newick": str(tree_path.relative_to(out_dir)),
                            "estimate_newick_sha256": sha256(tree_path),
                            "estimate_splits": splits,
                            **score,
                        }
                    )
                    records[(mode, seed)].append(score)
                stable_splits[mode] += len(set(seed_splits)) == 1
                stable_truth[mode] += len(set(seed_truth)) == 1

        assert canonical_active is not None and canonical_total is not None
        n_trees = len(tree_ids)
        for mode in scale_modes:
            for seed in seeds:
                summaries.append(
                    {
                        "suite_id": suite_id,
                        "metric": metric,
                        "scale_mode": mode,
                        "seed": seed,
                        "epochs": epochs,
                        "n_trees": n_trees,
                        "n_seed_stable_complete_split_sets": stable_splits[mode],
                        "seed_stable_complete_split_fraction": stable_splits[mode] / n_trees,
                        "n_seed_stable_truth_recovery": stable_truth[mode],
                        "seed_stable_truth_recovery_fraction": stable_truth[mode] / n_trees,
                        **aggregate_recovery(records[(mode, seed)]),
                    }
                )
        suite_audits.append(
            {
                "suite_id": suite_id,
                "analysis_root": str(root),
                "n_trees": n_trees,
                "n_tensors_total": canonical_total,
                "n_tensors_active": len(canonical_active),
                "n_numerical_negative_entries_clipped": n_clipped_entries,
                "minimum_distance_before_clipping": minimum_before_clipping,
                "active_tensor_names": canonical_active,
                "active_tensor_names_sha256": hashlib.sha256(
                    json.dumps(canonical_active, separators=(",", ":")).encode()
                ).hexdigest(),
                "input_files": input_files,
            }
        )

    payload = {
        "schema": SCHEMA,
        "valid": True,
        "estimator": {
            "layer_population": "trained layers: cosine-distance matrices above the zero tolerance",
            "layer_tree": "neighbor joining on one layer's distance matrix",
            "aggregate": "BHV Sturm Frechet mean over layer trees",
            "negative_pendant_edges": "clipped to zero before BHV aggregation",
            "numerical_negative_distances": "values in [-zero_tolerance, 0) clipped to zero",
        },
        "parameters": {
            "metric": metric,
            "scale_modes": list(scale_modes),
            "epochs": epochs,
            "seeds": list(seeds),
            "zero_tolerance": zero_tolerance,
            "branch_length_format": branch_length_format or "biopython default (%1.5f)",
            "tree_ids": list(tree_ids),
        },
        "suite_audits": suite_audits,
        "summaries": summaries,
        "observations": observations,
    }
    (out_dir / "bhv_table_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    with (out_dir / "bhv_table_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: row.get(key) for key in SUMMARY_COLUMNS})
    return payload
