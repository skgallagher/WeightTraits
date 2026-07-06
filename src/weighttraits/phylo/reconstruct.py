"""Neighbor-joining reconstruction from WeightTraits distance cubes."""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from weighttraits.distances.streaming import DistanceCube


@dataclass(frozen=True)
class ReconstructionResult:
    """A reconstructed tree plus the metadata needed to audit it."""

    newick: str
    audit: dict[str, Any]


def load_distance_cube(cube_dir: str | Path) -> DistanceCube:
    """Load a distance cube written by ``write_distance_cube``."""

    path = Path(cube_dir)
    npz_path = path / "distance_cube.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"missing distance cube: {npz_path}")

    with np.load(npz_path) as data:
        distances = {metric: np.asarray(data[metric], dtype=np.float64) for metric in data.files}

    layers = json.loads((path / "layers.json").read_text())
    models = json.loads((path / "models.json").read_text())
    audit_path = path / "audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    return DistanceCube(distances=distances, layer_names=layers, model_ids=models, audit=audit)


def matrix_from_distance_cube(
    cube: DistanceCube,
    *,
    metric: str,
    layer: str | int | None = None,
    aggregate: str = "mean",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select or aggregate one model-by-model matrix from a distance cube."""

    if metric not in cube.distances:
        raise ValueError(f"metric {metric!r} is not in cube; available={sorted(cube.distances)}")
    values = np.asarray(cube.distances[metric], dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"metric {metric!r} must have shape layer x model x model")

    if layer is not None:
        layer_index = _resolve_layer(layer, cube.layer_names)
        matrix = values[layer_index]
        selection = {
            "layer": cube.layer_names[layer_index],
            "layer_index": layer_index,
            "aggregate": None,
        }
    else:
        matrix = _aggregate_layers(values, aggregate)
        selection = {"layer": None, "layer_index": None, "aggregate": aggregate}

    matrix = _validated_distance_matrix(matrix, cube.model_ids)
    selection.update(_matrix_summary(matrix))
    return matrix, selection


def reconstruct_tree_from_cube(
    cube_dir: str | Path,
    *,
    metric: str,
    layer: str | int | None = None,
    aggregate: str = "mean",
) -> ReconstructionResult:
    """Reconstruct a neighbor-joining tree from a persisted distance cube."""

    cube = load_distance_cube(cube_dir)
    matrix, selection = matrix_from_distance_cube(
        cube,
        metric=metric,
        layer=layer,
        aggregate=aggregate,
    )
    newick = neighbor_joining_newick(cube.model_ids, matrix)
    audit = {
        "source_cube": str(cube_dir),
        "metric": metric,
        "n_models": len(cube.model_ids),
        "model_ids": cube.model_ids,
        **selection,
    }
    return ReconstructionResult(newick=newick, audit=audit)


def neighbor_joining_newick(labels: Sequence[str], distances: np.ndarray) -> str:
    """Return a Newick tree from a labeled distance matrix using Biopython NJ."""

    labels = list(labels)
    matrix = _validated_distance_matrix(distances, labels)
    phylo, distance_matrix, constructor = _require_biopython_tree_construction()
    lower_triangle = [
        [float(matrix[row, col]) for col in range(row + 1)]
        for row in range(len(labels))
    ]
    tree = constructor().nj(distance_matrix(labels, lower_triangle))
    out = StringIO()
    phylo.write(tree, out, "newick")
    return out.getvalue().strip()


def _resolve_layer(layer: str | int, layer_names: Sequence[str]) -> int:
    if isinstance(layer, int):
        index = layer
    else:
        try:
            index = int(layer)
        except ValueError:
            if layer not in layer_names:
                raise ValueError(f"layer {layer!r} is not in cube") from None
            index = list(layer_names).index(layer)
    if index < 0 or index >= len(layer_names):
        raise ValueError(f"layer index {index} is outside 0..{len(layer_names) - 1}")
    return index


def _aggregate_layers(values: np.ndarray, aggregate: str) -> np.ndarray:
    if aggregate == "mean":
        return np.mean(values, axis=0)
    if aggregate == "median":
        return np.median(values, axis=0)
    raise ValueError(f"unsupported layer aggregate: {aggregate}")


def _validated_distance_matrix(
    matrix: np.ndarray,
    labels: Sequence[str],
    *,
    tolerance: float = 1e-9,
) -> np.ndarray:
    out = np.asarray(matrix, dtype=np.float64)
    n = len(labels)
    if n < 2:
        raise ValueError("neighbor joining requires at least two labels")
    if len(set(labels)) != n:
        raise ValueError("distance matrix labels must be unique")
    if out.shape != (n, n):
        raise ValueError(f"distance matrix shape {out.shape} does not match {n} labels")
    if not np.all(np.isfinite(out)):
        raise ValueError("distance matrix contains non-finite values")
    if np.max(np.abs(out - out.T)) > tolerance:
        raise ValueError("distance matrix must be symmetric")
    if np.max(np.abs(np.diag(out))) > tolerance:
        raise ValueError("distance matrix diagonal must be zero")
    if np.min(out) < -tolerance:
        raise ValueError("distance matrix contains negative distances")

    out = (out + out.T) / 2.0
    out[np.abs(out) <= tolerance] = 0.0
    np.fill_diagonal(out, 0.0)
    return out


def _require_biopython_tree_construction():
    try:
        from Bio import Phylo
        from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
    except ImportError as exc:
        raise ImportError(
            "Biopython is required for neighbor-joining reconstruction; "
            "install WeightTraits with the analysis extra."
        ) from exc
    return Phylo, DistanceMatrix, DistanceTreeConstructor


def _matrix_summary(matrix: np.ndarray) -> dict[str, float]:
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    off_diag = matrix[mask]
    return {
        "distance_min": float(off_diag.min()) if off_diag.size else 0.0,
        "distance_max": float(off_diag.max()) if off_diag.size else 0.0,
        "distance_mean": float(off_diag.mean()) if off_diag.size else 0.0,
    }
