"""Atteson-margin diagnostics for neighbor-joining recovery.

Atteson's neighbor-joining guarantee is an all-edge bottleneck statement: if the
observed distance matrix is within half the shortest true tree edge length in
``l_infinity`` error, NJ returns the true topology. For empirical distances we fit
nonnegative additive edge lengths to a known topology, then report the same
edge-length / error ratio as an oracle diagnostic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from weighttraits.phylo.newick import NewickNode, leaf_names, parse_newick


@dataclass(frozen=True)
class AttesonEdgeMargin:
    """Per-edge fitted length and Atteson margin."""

    split: tuple[str, ...]
    length: float
    margin: float
    is_pendant: bool
    is_internal: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "split": list(self.split),
            "length": self.length,
            "margin": self.margin,
            "is_pendant": self.is_pendant,
            "is_internal": self.is_internal,
        }


@dataclass(frozen=True)
class AttesonMarginResult:
    """All-edge Atteson bottleneck plus internal-edge diagnostics."""

    labels: tuple[str, ...]
    error_linf: float
    denominator: float
    n_edges: int
    n_pendant_edges: int
    n_internal_edges: int
    bottleneck_margin: float
    internal_bottleneck_margin: float | None
    mean_internal_margin: float | None
    frac_internal_resolvable: float | None
    theorem_certified: bool
    edges: tuple[AttesonEdgeMargin, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "labels": list(self.labels),
            "error_linf": self.error_linf,
            "denominator": self.denominator,
            "n_edges": self.n_edges,
            "n_pendant_edges": self.n_pendant_edges,
            "n_internal_edges": self.n_internal_edges,
            "bottleneck_margin": self.bottleneck_margin,
            "internal_bottleneck_margin": self.internal_bottleneck_margin,
            "mean_internal_margin": self.mean_internal_margin,
            "frac_internal_resolvable": self.frac_internal_resolvable,
            "theorem_certified": self.theorem_certified,
            "edges": [edge.as_dict() for edge in self.edges],
        }


def atteson_margin(
    labels: Sequence[str],
    distances: np.ndarray,
    truth_newick: str | NewickNode,
    *,
    eps: float = 1e-12,
) -> AttesonMarginResult:
    """Fit an additive metric on ``truth_newick`` and compute Atteson margins.

    The returned ``bottleneck_margin`` is the minimum over all fitted tree edges,
    including pendant edges. ``internal_bottleneck_margin`` and
    ``mean_internal_margin`` are topology-edge diagnostics only; they are not the
    Atteson theorem certificate.
    """

    label_tuple = tuple(labels)
    matrix = _validated_distance_matrix(distances, label_tuple)
    root = parse_newick(truth_newick) if isinstance(truth_newick, str) else truth_newick
    splits = edge_splits(root, label_tuple)
    if not splits:
        raise ValueError("truth tree has no usable edges for the requested labels")

    pair_indices = [(i, j) for i in range(len(label_tuple)) for j in range(i + 1, len(label_tuple))]
    design = np.zeros((len(pair_indices), len(splits)), dtype=float)
    for row, (i, j) in enumerate(pair_indices):
        left = label_tuple[i]
        right = label_tuple[j]
        for col, split in enumerate(splits):
            if (left in split) ^ (right in split):
                design[row, col] = 1.0

    observed = np.array([matrix[i, j] for i, j in pair_indices], dtype=float)
    lengths = _nnls(design, observed)
    fitted = design @ lengths
    error_linf = float(np.max(np.abs(fitted - observed))) if observed.size else 0.0
    denominator = max(2.0 * error_linf, eps)

    edges: list[AttesonEdgeMargin] = []
    n_leaves = len(label_tuple)
    for split, length in zip(splits, lengths, strict=True):
        split_size = len(split)
        is_pendant = split_size == 1 or split_size == n_leaves - 1
        is_internal = 2 <= split_size <= n_leaves - 2
        edges.append(
            AttesonEdgeMargin(
                split=tuple(sorted(split)),
                length=float(length),
                margin=float(length / denominator),
                is_pendant=is_pendant,
                is_internal=is_internal,
            )
        )

    all_margins = [edge.margin for edge in edges]
    internal_margins = [edge.margin for edge in edges if edge.is_internal]
    bottleneck = min(all_margins, default=math.inf)
    return AttesonMarginResult(
        labels=label_tuple,
        error_linf=error_linf,
        denominator=denominator,
        n_edges=len(edges),
        n_pendant_edges=sum(edge.is_pendant for edge in edges),
        n_internal_edges=sum(edge.is_internal for edge in edges),
        bottleneck_margin=bottleneck,
        internal_bottleneck_margin=min(internal_margins) if internal_margins else None,
        mean_internal_margin=float(np.mean(internal_margins)) if internal_margins else None,
        frac_internal_resolvable=(
            float(np.mean(np.asarray(internal_margins) > 1.0)) if internal_margins else None
        ),
        theorem_certified=bottleneck > 1.0,
        edges=tuple(edges),
    )


def edge_splits(root: NewickNode, labels: Sequence[str]) -> tuple[frozenset[str], ...]:
    """Return canonical splits for every tree edge, pendant edges included."""

    requested = frozenset(labels)
    if len(requested) != len(labels):
        raise ValueError("labels must be unique")
    tree_leaves = set(leaf_names(root))
    missing = requested - tree_leaves
    if missing:
        raise ValueError(f"truth tree is missing labels: {sorted(missing)}")

    splits: list[frozenset[str]] = []
    seen: set[frozenset[str]] = set()

    def add_split(descendants: frozenset[str]) -> None:
        split = _canonical_edge_split(descendants & requested, requested)
        if split is not None and split not in seen:
            seen.add(split)
            splits.append(split)

    def visit(node: NewickNode) -> frozenset[str]:
        if node.is_leaf:
            return frozenset([node.name]) if node.name in requested else frozenset()
        descendants: set[str] = set()
        for child in node.children:
            child_descendants = visit(child)
            add_split(child_descendants)
            descendants.update(child_descendants)
        return frozenset(descendants)

    visit(root)
    return tuple(sorted(splits, key=lambda split: (len(split), tuple(sorted(split)))))


def _canonical_edge_split(
    side: frozenset[str],
    all_leaves: frozenset[str],
) -> frozenset[str] | None:
    other = all_leaves - side
    if not side or not other:
        return None
    if len(side) < len(other):
        return side
    if len(other) < len(side):
        return other
    return min(side, other, key=lambda values: tuple(sorted(values)))


def _validated_distance_matrix(matrix: np.ndarray, labels: Sequence[str]) -> np.ndarray:
    out = np.asarray(matrix, dtype=float)
    n = len(labels)
    if n < 2:
        raise ValueError("Atteson margin requires at least two labels")
    if len(set(labels)) != n:
        raise ValueError("distance matrix labels must be unique")
    if out.shape != (n, n):
        raise ValueError(f"distance matrix shape {out.shape} does not match {n} labels")
    if not np.all(np.isfinite(out)):
        raise ValueError("distance matrix contains non-finite values")
    if np.max(np.abs(out - out.T)) > 1e-9:
        raise ValueError("distance matrix must be symmetric")
    if np.max(np.abs(np.diag(out))) > 1e-9:
        raise ValueError("distance matrix diagonal must be zero")
    if np.min(out) < -1e-9:
        raise ValueError("distance matrix contains negative distances")
    out = (out + out.T) / 2.0
    np.fill_diagonal(out, 0.0)
    return out


def _nnls(design: np.ndarray, observed: np.ndarray) -> np.ndarray:
    try:
        from scipy.optimize import nnls
    except ImportError as exc:
        raise ImportError(
            "SciPy is required for Atteson margin fitting; install WeightTraits "
            "with the analysis extra."
        ) from exc
    lengths, _ = nnls(design, observed)
    return np.asarray(lengths, dtype=float)
