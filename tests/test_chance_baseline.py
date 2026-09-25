import glob
from fractions import Fraction
from itertools import combinations
from pathlib import Path

import pytest

from weighttraits.phylo.chance import (
    binary_refinement_count,
    chance_baseline_from_manifests,
    chance_clade_estimation,
    chance_full_estimation,
    double_factorial,
    internal_vertex_degrees,
    split_probability,
    unrooted_binary_topology_count,
)

ROOT = Path(__file__).parents[1]
MANIFESTS = ROOT / "examples" / "training" / "confirm_paper_numbers" / "assigned_manifests"


def _all_unrooted_binary_split_sets(leaves):
    """Enumerate every unrooted binary tree by stepwise leaf addition; return its split sets."""

    leaves = sorted(leaves)
    first, second, third = leaves[:3]
    trees = [[("c", first), ("c", second), ("c", third)]]
    for index, leaf in enumerate(leaves[3:]):
        grown = []
        for edges in trees:
            for position, (u, v) in enumerate(edges):
                hub = f"h{index}"
                rest = edges[:position] + edges[position + 1 :]
                grown.append(rest + [(u, hub), (hub, v), (hub, leaf)])
        trees = grown
    split_sets = []
    for edges in trees:
        adjacency = {}
        for u, v in edges:
            adjacency.setdefault(u, set()).add(v)
            adjacency.setdefault(v, set()).add(u)
        splits = set()
        for u, v in edges:
            seen, stack = {u}, [v]
            while stack:
                node = stack.pop()
                if node in seen:
                    continue
                seen.add(node)
                stack.extend(adjacency[node] - seen)
            side = frozenset(node for node in seen - {u} if node in leaves)
            if 1 < len(side) < len(leaves) - 1:
                splits.add(min(side, frozenset(leaves) - side, key=lambda s: (len(s), sorted(s))))
        split_sets.append(splits)
    return split_sets


def test_double_factorial_and_topology_counts():
    assert [double_factorial(k) for k in (-1, 0, 1, 3, 5, 7)] == [1, 1, 1, 3, 15, 105]
    assert [unrooted_binary_topology_count(n) for n in (3, 4, 5, 6, 10)] == [1, 3, 15, 105, 2027025]


@pytest.mark.parametrize("n_leaves", [4, 5, 6, 7])
def test_split_probability_matches_enumeration(n_leaves):
    leaves = [f"x{i}" for i in range(n_leaves)]
    trees = _all_unrooted_binary_split_sets(leaves)
    assert len(trees) == unrooted_binary_topology_count(n_leaves)
    for size in range(2, n_leaves // 2 + 1):
        split = frozenset(leaves[:size])
        canonical = min(split, frozenset(leaves) - split, key=lambda s: (len(s), sorted(s)))
        hits = sum(canonical in splits for splits in trees)
        assert Fraction(hits, len(trees)) == split_probability(n_leaves, size)


def test_refinement_count_matches_enumeration_for_multifurcating_trees():
    leaves = [f"x{i}" for i in range(7)]
    trees = _all_unrooted_binary_split_sets(leaves)
    cases = [
        set(),
        {frozenset({"x0", "x1"})},
        {frozenset({"x0", "x1"}), frozenset({"x0", "x1", "x2"})},
        {frozenset({"x0", "x1", "x2"}), frozenset({"x3", "x4"})},
    ]
    for truth in cases:
        hits = sum(truth <= splits for splits in trees)
        assert hits == binary_refinement_count(truth, leaves)
        if truth:
            assert chance_full_estimation(sorted(truth, key=sorted), leaves) == Fraction(
                hits, len(trees)
            )


def test_degrees_of_a_star_and_rejection_of_incompatible_splits():
    leaves = [f"x{i}" for i in range(6)]
    assert internal_vertex_degrees([], leaves) == [6]
    assert internal_vertex_degrees([frozenset({"x0", "x1"})], leaves) == [3, 5]
    with pytest.raises(ValueError, match="compatible"):
        internal_vertex_degrees([frozenset({"x0", "x1"}), frozenset({"x1", "x2"})], leaves)


def test_quartet_clade_chance_is_one_third():
    assert chance_clade_estimation([frozenset({"a", "b"})], "abcd") == Fraction(1, 3)


@pytest.mark.skipif(not MANIFESTS.is_dir(), reason="paper truth manifests not present")
def test_paper_chance_baseline_on_the_46_eligible_trees():
    result = chance_baseline_from_manifests(glob.glob(str(MANIFESTS / "*.manifest.jsonl")))

    assert result["n_trees"] == 46
    assert result["skipped_no_clade"] == [
        f"confirm_paper_tree_{index:03d}" for index in (15, 20, 37, 47)
    ]
    assert round(100 * result["clade_estimation"]["mean"], 1) == 8.7
    assert round(100 * result["clade_estimation"]["se"], 1) == 1.1
    assert round(100 * result["full_estimation"]["mean"], 1) == 3.2


def test_every_pair_of_small_splits_is_scored_once():
    # sanity check on the helper used above: 7 leaves have C(7,2) cherry candidates
    leaves = [f"x{i}" for i in range(7)]
    trees = _all_unrooted_binary_split_sets(leaves)
    for pair in combinations(leaves, 2):
        hits = sum(frozenset(pair) in splits for splits in trees)
        assert Fraction(hits, len(trees)) == split_probability(7, 2)
