import math

import numpy as np
import pytest

pytest.importorskip("scipy")

from weighttraits.phylo.atteson import atteson_margin, edge_splits
from weighttraits.phylo.newick import parse_newick


LABELS = ("a", "b", "c", "d")
TRUTH = "((a,b),(c,d));"


def _matrix_from_edge_lengths(edge_lengths: dict[frozenset[str], float]) -> np.ndarray:
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=float)
    for split, length in edge_lengths.items():
        for i, left in enumerate(LABELS):
            for j, right in enumerate(LABELS):
                if i < j and ((left in split) ^ (right in split)):
                    matrix[i, j] += length
                    matrix[j, i] += length
    return matrix


def test_edge_splits_include_pendant_and_internal_edges():
    splits = edge_splits(parse_newick(TRUTH), LABELS)

    assert len(splits) == 5
    assert frozenset({"a"}) in splits
    assert frozenset({"b"}) in splits
    assert frozenset({"c"}) in splits
    assert frozenset({"d"}) in splits
    assert frozenset({"a", "b"}) in splits


def test_atteson_margin_fits_exact_additive_tree_on_all_edges():
    distances = _matrix_from_edge_lengths(
        {
            frozenset({"a"}): 1.0,
            frozenset({"b"}): 1.0,
            frozenset({"c"}): 1.0,
            frozenset({"d"}): 1.0,
            frozenset({"a", "b"}): 2.0,
        }
    )

    result = atteson_margin(LABELS, distances, TRUTH)

    assert result.n_edges == 5
    assert result.n_pendant_edges == 4
    assert result.n_internal_edges == 1
    assert result.error_linf < 1e-10
    assert result.bottleneck_margin > 1e11
    assert result.theorem_certified
    assert sorted(round(edge.length, 6) for edge in result.edges) == [1.0, 1.0, 1.0, 1.0, 2.0]


def test_all_edge_bottleneck_can_be_pendant_when_internal_margin_passes():
    distances = _matrix_from_edge_lengths(
        {
            frozenset({"a"}): 0.01,
            frozenset({"b"}): 2.0,
            frozenset({"c"}): 2.0,
            frozenset({"d"}): 2.0,
            frozenset({"a", "b"}): 3.0,
        }
    )
    noisy = distances.copy()
    noisy[0, 2] += 0.18
    noisy[2, 0] += 0.18

    result = atteson_margin(LABELS, noisy, TRUTH)
    bottleneck_edge = min(result.edges, key=lambda edge: edge.margin)

    assert bottleneck_edge.is_pendant
    assert bottleneck_edge.split == ("a",)
    assert result.internal_bottleneck_margin is not None
    assert result.internal_bottleneck_margin > 1.0
    assert result.bottleneck_margin < 1.0
    assert not result.theorem_certified


def test_atteson_margin_rejects_missing_truth_labels():
    distances = np.zeros((2, 2), dtype=float)

    with pytest.raises(ValueError, match="missing labels"):
        atteson_margin(("a", "z"), distances, TRUTH)


def test_atteson_margin_dict_is_json_ready():
    distances = _matrix_from_edge_lengths(
        {
            frozenset({"a"}): 1.0,
            frozenset({"b"}): 1.0,
            frozenset({"c"}): 1.0,
            frozenset({"d"}): 1.0,
            frozenset({"a", "b"}): 2.0,
        }
    )

    payload = atteson_margin(LABELS, distances, TRUTH).as_dict()

    assert payload["n_edges"] == 5
    assert math.isfinite(payload["edges"][0]["length"])
