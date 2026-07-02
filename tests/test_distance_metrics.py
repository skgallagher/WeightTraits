import math

import numpy as np

from weighttraits.distances.metrics import (
    correlation_distance,
    cosine_distance,
    l1_distance,
    l2_distance,
    pairwise_distance_matrix,
    threshold_distance,
)


def test_basic_distances_match_known_values():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])

    assert math.isclose(cosine_distance(a, b), 1.0)
    assert math.isclose(l1_distance(a, b), 2.0)
    assert math.isclose(l2_distance(a, b), math.sqrt(2.0))
    assert math.isclose(threshold_distance(a, b, eps=0.5), 2.0)


def test_degenerate_cosine_and_correlation_are_zero():
    assert cosine_distance([0, 0], [1, 2]) == 0.0
    assert correlation_distance([1, 1], [2, 3]) == 0.0


def test_pairwise_distance_matrix_is_symmetric_with_zero_diagonal():
    matrix = pairwise_distance_matrix([[1, 0], [0, 1], [1, 1]], metric=cosine_distance)
    np.testing.assert_allclose(matrix, matrix.T)
    np.testing.assert_allclose(np.diag(matrix), np.zeros(3))
    assert matrix.shape == (3, 3)

