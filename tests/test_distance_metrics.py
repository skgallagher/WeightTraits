import math

import numpy as np

from weighttraits.distances.metrics import (
    available_metrics,
    correlation_distance,
    cosine_distance,
    get_metric,
    l1_distance,
    l2_distance,
    linear_cka_distance,
    linear_cka_similarity,
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


def test_metric_registry_exposes_named_metrics():
    assert "cosine" in available_metrics()
    assert "cka" in available_metrics()
    assert get_metric("cosine") is cosine_distance


def test_pairwise_distance_matrix_accepts_metric_name_and_kwargs():
    matrix = pairwise_distance_matrix([[0.0, 1.0], [0.2, 1.2]], metric="threshold", eps=0.1)
    np.testing.assert_allclose(matrix, np.array([[0.0, 2.0], [2.0, 0.0]]))


def test_linear_cka_identical_and_scaled_matrices_have_zero_distance():
    x = np.array(
        [
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 1.0],
            [2.0, 1.0, 0.0],
            [3.0, 2.0, 1.0],
        ]
    )

    assert math.isclose(linear_cka_similarity(x, x), 1.0)
    assert math.isclose(linear_cka_distance(x, 3.0 * x), 0.0, abs_tol=1e-12)


def test_linear_cka_is_invariant_to_orthogonal_feature_rotation():
    x = np.array(
        [
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 1.0],
            [2.0, 1.0, 0.0],
            [3.0, 2.0, 1.0],
        ]
    )
    q = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )

    assert math.isclose(linear_cka_distance(x, x @ q), 0.0, abs_tol=1e-12)


def test_pairwise_cka_distance_matrix_uses_matrix_inputs():
    x = np.arange(12, dtype=float).reshape(4, 3)
    y = np.flipud(x)
    z = 2.0 * x

    matrix = pairwise_distance_matrix([x, y, z], metric="cka")

    np.testing.assert_allclose(np.diag(matrix), np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(matrix, matrix.T)
    assert math.isclose(matrix[0, 2], 0.0, abs_tol=1e-12)
    assert matrix[0, 1] >= 0.0
