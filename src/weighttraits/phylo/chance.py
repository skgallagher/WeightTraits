"""Exact chance baselines for clade estimation and full estimation.

A "random tree" is a uniformly random unrooted binary topology on the observed leaves. There
are ``(2n-5)!!`` of them on ``n`` labelled leaves. A fixed split with sides of size ``a`` and
``b = n - a`` appears in ``(2a-3)!! (2b-3)!!`` of them: contracting the split edge leaves a
rooted binary tree on each side, and there are ``(2k-3)!!`` rooted binary trees on ``k`` leaves.

A random tree contains every clade of a (possibly multifurcating) true tree only if it refines
that tree. An unrooted tree whose internal vertices have degrees ``d_v`` has
``prod_v (2 d_v - 5)!!`` binary refinements.

Both baselines are exact expected values, computed with rational arithmetic. They are
averaged over trees the same way the paper averages clade and full estimation.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from weighttraits.phylo.splits import Split, splits_from_manifest_path


def double_factorial(k: int) -> int:
    """``k!!`` with the convention ``(-1)!! = 0!! = 1``."""

    if k < -1:
        raise ValueError(f"double factorial undefined for {k}")
    return math.prod(range(k, 0, -2)) if k > 0 else 1


def unrooted_binary_topology_count(n_leaves: int) -> int:
    """Number of unrooted binary topologies on ``n_leaves`` labelled leaves (``n >= 3``)."""

    if n_leaves < 3:
        raise ValueError("an unrooted binary tree needs at least three leaves")
    return double_factorial(2 * n_leaves - 5)


def split_probability(n_leaves: int, side_size: int) -> Fraction:
    """Probability that a uniformly random unrooted binary tree contains a given split."""

    other = n_leaves - side_size
    if side_size < 1 or other < 1:
        raise ValueError(f"split side {side_size} is invalid for {n_leaves} leaves")
    numerator = double_factorial(2 * side_size - 3) * double_factorial(2 * other - 3)
    return Fraction(numerator, unrooted_binary_topology_count(n_leaves))


def internal_vertex_degrees(splits: Iterable[Split], leaves: Iterable[str]) -> list[int]:
    """Degrees of the internal vertices of the unrooted tree defined by a compatible split set.

    The tree is rooted at an arbitrary reference leaf ``r``. Every split then has one side
    without ``r`` (a cluster), and the clusters plus ``L - {r}`` form a hierarchy. Each cluster is
    one internal vertex whose degree is its number of children plus one parent edge.
    """

    leaf_set = frozenset(leaves)
    if len(leaf_set) < 3:
        raise ValueError("need at least three leaves")
    reference = min(leaf_set)
    clusters = {leaf_set - {reference}}
    for split in splits:
        side = frozenset(split)
        cluster = leaf_set - side if reference in side else side
        if not 1 < len(cluster) < len(leaf_set) - 1:
            raise ValueError(f"split {sorted(side)} is not a nontrivial bipartition")
        clusters.add(cluster)
    ordered = sorted(clusters, key=len)
    for i, small in enumerate(ordered):
        for large in ordered[i + 1 :]:
            if not (small <= large or small.isdisjoint(large)):
                raise ValueError("splits are not pairwise compatible")
    degrees = []
    for cluster in clusters:
        maximal = [
            sub
            for sub in clusters
            if sub < cluster and not any(sub < mid < cluster for mid in clusters)
        ]
        covered = frozenset().union(*maximal) if maximal else frozenset()
        n_children = len(maximal) + len(cluster - covered)
        degrees.append(n_children + 1)
    return sorted(degrees)


def binary_refinement_count(splits: Iterable[Split], leaves: Iterable[str]) -> int:
    """Number of unrooted binary trees that contain every split in ``splits``."""

    return math.prod(double_factorial(2 * d - 5) for d in internal_vertex_degrees(splits, leaves))


def chance_clade_estimation(splits: Sequence[Split], leaves: Iterable[str]) -> Fraction:
    """Expected fraction of the true clades that a random binary tree contains."""

    if not splits:
        raise ValueError("a tree with no nontrivial split has no clade to estimate")
    n_leaves = len(frozenset(leaves))
    probabilities = [split_probability(n_leaves, len(split)) for split in splits]
    return sum(probabilities, Fraction(0)) / len(probabilities)


def chance_full_estimation(splits: Sequence[Split], leaves: Iterable[str]) -> Fraction:
    """Probability that a random binary tree contains every true clade (``FN = 0``)."""

    leaf_set = frozenset(leaves)
    return Fraction(
        binary_refinement_count(splits, leaf_set),
        unrooted_binary_topology_count(len(leaf_set)),
    )


def chance_baseline_from_manifests(paths: Iterable[str | Path]) -> dict[str, Any]:
    """Per-tree and averaged chance baselines over truth manifests.

    Trees without a nontrivial split are skipped (they have no clade to estimate), mirroring
    the paper's topology-eligible set. Averages are unweighted means over trees; the SE is the
    sample standard deviation across trees divided by ``sqrt(n)``.
    """

    rows = []
    skipped = []
    for path in sorted(Path(p) for p in paths):
        splits, leaves = splits_from_manifest_path(str(path))
        tree_id = path.name.split(".")[0]
        if not splits:
            skipped.append(tree_id)
            continue
        ordered = sorted(splits, key=lambda split: (len(split), sorted(split)))
        rows.append(
            {
                "tree_id": tree_id,
                "n_leaves": len(leaves),
                "n_true_clades": len(ordered),
                "chance_clade_estimation": float(chance_clade_estimation(ordered, leaves)),
                "chance_full_estimation": float(chance_full_estimation(ordered, leaves)),
            }
        )
    if not rows:
        raise ValueError("no manifest has a clade to estimate")

    def summary(key: str) -> dict[str, float]:
        values = [row[key] for row in rows]
        se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
        return {"mean": statistics.fmean(values), "se": se}

    return {
        "schema": "weighttraits.chance_baseline.v1",
        "random_tree": "uniform over unrooted binary topologies on the observed leaves",
        "n_trees": len(rows),
        "skipped_no_clade": skipped,
        "clade_estimation": summary("chance_clade_estimation"),
        "full_estimation": summary("chance_full_estimation"),
        "trees": rows,
    }
